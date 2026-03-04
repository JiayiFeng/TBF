from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

from .format import (
    DEFAULT_PAGE_SIZE,
    FILE_HEADER_STRUCT,
    FILE_MAGIC,
    FOOTER_MAGIC,
    FOOTER_STRUCT,
    INDEX_ENTRY_PREFIX_STRUCT,
    INDEX_HEADER_STRUCT,
    INDEX_MAGIC,
    VERSION,
    align_up,
    torch_dtype_maps,
)


@dataclass
class _IndexEntry:
    record_id: int
    key: bytes
    dtype_code: int
    shape: list[int]
    data_offset: int
    nbytes: int


class TBFWriter:
    """Streaming writer for Tensor Batch Format."""

    def __init__(self, path: str | Path, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self.path = str(path)
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise ValueError("page_size must be > 0")

        self._f = open(self.path, "wb")
        self._f.write(FILE_HEADER_STRUCT.pack(FILE_MAGIC, VERSION, 0))
        self._entries: list[_IndexEntry] = []
        self._record_count = 0
        self._closed = False
        self._dtype_to_code, _, _ = torch_dtype_maps()

    def close(self) -> None:
        if self._closed:
            return
        self._write_index_and_footer()
        self._f.close()
        self._closed = True

    def __enter__(self) -> "TBFWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def add_record(self, record: dict[str, object]) -> None:
        record_id = self._record_count
        for key, tensor in record.items():
            self.add_tensor(record_id, key, tensor)
        self._record_count += 1

    def add_records(self, records: list[dict[str, object]]) -> None:
        for record in records:
            self.add_record(record)

    def add_tensor(self, record_id: int, key: str, tensor: object) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"value for key '{key}' is not a torch.Tensor")

        t = tensor.detach().cpu().contiguous()
        dtype_code = self._dtype_to_code.get(t.dtype)
        if dtype_code is None:
            raise TypeError(f"unsupported dtype: {t.dtype}")

        nbytes = int(t.numel() * t.element_size())
        aligned_offset = align_up(self._f.tell(), self.page_size)
        pad_len = aligned_offset - self._f.tell()
        if pad_len:
            self._f.write(b"\x00" * pad_len)

        if nbytes > 0:
            self._f.write(memoryview(t.numpy()))
        

        self._entries.append(
            _IndexEntry(
                record_id=record_id,
                key=key.encode("utf-8"),
                dtype_code=dtype_code,
                shape=[int(x) for x in t.shape],
                data_offset=aligned_offset,
                nbytes=nbytes,
            )
        )

    def _write_index_and_footer(self) -> None:
        index_offset = self._f.tell()
        self._f.write(
            INDEX_HEADER_STRUCT.pack(
                INDEX_MAGIC,
                VERSION,
                len(self._entries),
                self._record_count,
            )
        )

        for entry in self._entries:
            self._f.write(
                INDEX_ENTRY_PREFIX_STRUCT.pack(
                    entry.record_id,
                    len(entry.key),
                    entry.dtype_code,
                    len(entry.shape),
                    entry.data_offset,
                    entry.nbytes,
                )
            )
            for dim in entry.shape:
                self._f.write(int(dim).to_bytes(8, byteorder="little", signed=True))
            self._f.write(entry.key)

        index_size = self._f.tell() - index_offset
        self._f.write(
            FOOTER_STRUCT.pack(
                FOOTER_MAGIC,
                VERSION,
                index_offset,
                index_size,
                b"\x00" * 36,
            )
        )


def write_tbf(path: str | Path, records: list[dict[str, object]], page_size: int = DEFAULT_PAGE_SIZE) -> None:
    with TBFWriter(path=path, page_size=page_size) as writer:
        writer.add_records(records)
