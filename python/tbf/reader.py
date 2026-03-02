from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path
import torch

from .format import (
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FOOTER_MAGIC,
    FOOTER_STRUCT,
    INDEX_ENTRY_PREFIX_STRUCT,
    INDEX_HEADER_STRUCT,
    INDEX_MAGIC,
    VERSION,
    torch_dtype_maps,
)


@dataclass
class TensorMeta:
    record_id: int
    key: str
    dtype_code: int
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int


class TBFReader:
    def __init__(self, path: str | Path):
        self.path = str(path)
        _, self._code_to_dtype, self._code_to_elsize = torch_dtype_maps()

        self._f = open(self.path, "rb")
        self._size = Path(self.path).stat().st_size
        if self._size < FILE_HEADER_STRUCT.size + FOOTER_STRUCT.size:
            self._f.close()
            raise ValueError("file too small")

        self._mmap = mmap.mmap(self._f.fileno(), length=0, access=mmap.ACCESS_READ)
        self.record_count: int
        self.entry_count: int
        self._entries: list[TensorMeta]
        self.record_count, self.entry_count, self._entries = self._parse_metadata()
        self._entries_by_record: list[list[TensorMeta]] = [[] for _ in range(self.record_count)]
        for entry in self._entries:
            self._entries_by_record[entry.record_id].append(entry)

    def close(self) -> None:
        if hasattr(self, "_mmap") and self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, "_f") and self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "TBFReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return self.record_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.record_count
        if index < 0 or index >= self.record_count:
            raise IndexError(index)
        out: dict[str, torch.Tensor] = {}
        for entry in self._entries_by_record[index]:
            dtype = self._code_to_dtype.get(entry.dtype_code)
            if dtype is None:
                raise ValueError(f"unknown dtype_code: {entry.dtype_code}")

            if entry.nbytes == 0:
                tensor = torch.empty(entry.shape, dtype=dtype)
            else:
                element_size = self._code_to_elsize[entry.dtype_code]
                count = entry.nbytes // element_size
                base = torch.frombuffer(self._mmap, dtype=dtype, count=count, offset=entry.data_offset)
                # Materialize tensor data so returned tensors do not depend on reader lifetime.
                tensor = base.view(entry.shape).clone()
            out[entry.key] = tensor
        return out

    def read_all(self) -> list[dict[str, torch.Tensor]]:
        """Read all records in a single bulk copy.

        Instead of cloning each tensor individually, the entire file is copied
        into a bytearray once.  Tensors are then created as views into that
        buffer (no extra per-tensor allocation/copy).  The bytearray is kept
        alive via Python's reference counting as long as any returned tensor
        is still referenced.
        """
        self._mmap.seek(0)
        raw = bytearray(self._mmap.read())
        result: list[dict[str, torch.Tensor]] = []
        for record_entries in self._entries_by_record:
            record: dict[str, torch.Tensor] = {}
            for entry in record_entries:
                dtype = self._code_to_dtype.get(entry.dtype_code)
                if dtype is None:
                    raise ValueError(f"unknown dtype_code: {entry.dtype_code}")
                if entry.nbytes == 0:
                    tensor = torch.empty(entry.shape, dtype=dtype)
                else:
                    element_size = self._code_to_elsize[entry.dtype_code]
                    count = entry.nbytes // element_size
                    # frombuffer holds a reference to raw; no clone needed.
                    tensor = torch.frombuffer(raw, dtype=dtype, count=count, offset=entry.data_offset).view(entry.shape)
                record[entry.key] = tensor
            result.append(record)
        return result

    def metadata(self) -> list[TensorMeta]:
        return list(self._entries)

    def _parse_metadata(self):
        header = self._mmap[: FILE_HEADER_STRUCT.size]
        magic, version, _ = FILE_HEADER_STRUCT.unpack(header)
        if magic != FILE_MAGIC:
            raise ValueError("invalid file magic")
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")

        footer_start = self._size - FOOTER_STRUCT.size
        footer = self._mmap[footer_start : footer_start + FOOTER_STRUCT.size]
        (
            footer_magic,
            footer_version,
            index_offset,
            index_size,
            _,
        ) = FOOTER_STRUCT.unpack(footer)

        if footer_magic != FOOTER_MAGIC:
            raise ValueError("invalid footer magic")
        if footer_version != VERSION:
            raise ValueError(f"unsupported footer version: {footer_version}")
        if index_offset + index_size > footer_start:
            raise ValueError("index points outside payload region")

        idx = self._mmap[index_offset : index_offset + index_size]
        if len(idx) < INDEX_HEADER_STRUCT.size:
            raise ValueError("truncated index")

        (
            index_magic,
            index_version,
            index_entry_count,
            index_record_count,
        ) = INDEX_HEADER_STRUCT.unpack(idx[: INDEX_HEADER_STRUCT.size])

        if index_magic != INDEX_MAGIC:
            raise ValueError("invalid index magic")
        if index_version != VERSION:
            raise ValueError(f"unsupported index version: {index_version}")

        entries: list[TensorMeta] = []
        pos = INDEX_HEADER_STRUCT.size
        for _ in range(index_entry_count):
            end_prefix = pos + INDEX_ENTRY_PREFIX_STRUCT.size
            if end_prefix > len(idx):
                raise ValueError("truncated index entry")
            (record_id, key_len, dtype_code, ndim, data_offset, nbytes) = INDEX_ENTRY_PREFIX_STRUCT.unpack(idx[pos:end_prefix])
            pos = end_prefix

            shape: list[int] = []
            for _ in range(ndim):
                end_dim = pos + 8
                if end_dim > len(idx):
                    raise ValueError("truncated index shape")
                shape.append(int.from_bytes(idx[pos:end_dim], byteorder="little", signed=True))
                pos = end_dim

            end_key = pos + key_len
            if end_key > len(idx):
                raise ValueError("truncated index key")
            key = idx[pos:end_key].decode("utf-8")
            pos = end_key

            entries.append(
                TensorMeta(
                    record_id=record_id,
                    key=key,
                    dtype_code=dtype_code,
                    shape=tuple(shape),
                    data_offset=data_offset,
                    nbytes=nbytes,
                )
            )

        if pos != len(idx):
            raise ValueError("unexpected trailing bytes in index")

        return index_record_count, index_entry_count, entries
