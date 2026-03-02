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
        try:
            self._mmap.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
        self._mv = memoryview(self._mmap)
        self.record_count: int
        self.entry_count: int
        self._entries: list[TensorMeta]
        self.record_count, self.entry_count, self._entries = self._parse_metadata()
        self._entries_by_record: list[list[TensorMeta]] = [[] for _ in range(self.record_count)]
        for entry in self._entries:
            self._entries_by_record[entry.record_id].append(entry)

    def close(self) -> None:
        # Drop references only; do not call mmap.close() so that any tensors
        # created via __getitem__ (which hold memoryview slices into the mmap)
        # remain valid.  The mmap is released by the GC when all such tensors
        # are collected.
        if hasattr(self, "_mv"):
            self._mv = None
        if hasattr(self, "_mmap"):
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
                # Zero-copy: memoryview slice is O(1); frombuffer creates a
                # typed tensor view into the mmap with no data copy.
                # The slice holds a reference to the mmap, keeping it alive
                # beyond reader.close().
                tensor = torch.frombuffer(
                    self._mv[entry.data_offset : entry.data_offset + entry.nbytes],
                    dtype=dtype,
                ).view(entry.shape)
            out[entry.key] = tensor
        return out

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
