from __future__ import annotations

import copy
import json
import os
import shutil
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .format import DEFAULT_PAGE_SIZE
from .writer import write_tbf


class _NotReadyError(RuntimeError):
    pass


@dataclass
class _ServerState:
    window_start_batch_id: int
    fetched_current_by_rank: list[bool]
    current_batch_by_rank: list[int]


class TBFBatchHTTPServer:
    def __init__(
        self,
        dataloader_start_at_batch_id: Callable[[int], Any],
        to_records_funcs: list[list[Callable[[Any], list[dict[str, Any]]]]],
        rank_ap_mapping: list[int],
        rank_ring_attn_mapping: list[int],
        prefetch_count: int,
        local_rank_count: int,
        local_dir: str | Path,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        if prefetch_count <= 0:
            raise ValueError("prefetch_count must be > 0")
        if local_rank_count <= 0:
            raise ValueError("local_rank_count must be > 0")

        self.dataloader_start_at_batch_id = dataloader_start_at_batch_id
        self.to_records_funcs = to_records_funcs
        self.rank_ap_mapping = rank_ap_mapping
        self.rank_ring_attn_mapping = rank_ring_attn_mapping
        self.prefetch_count = prefetch_count
        self.local_rank_count = local_rank_count
        self.page_size = page_size

        self.local_dir = Path(local_dir)
        if self.local_dir.exists():
            shutil.rmtree(self.local_dir)
        self.local_dir.mkdir(parents=True)
        self.shared_dir = self.local_dir / "shared"
        self.rank_dirs = [self.local_dir / f"rank_{r}" for r in range(local_rank_count)]
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        for rank_dir in self.rank_dirs:
            rank_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._state = _ServerState(
            window_start_batch_id=0,
            fetched_current_by_rank=[False for _ in range(local_rank_count)],
            current_batch_by_rank=[-1 for _ in range(local_rank_count)],
        )

        self._dataloader = self.dataloader_start_at_batch_id(0)
        self._dataloader_iter = iter(self._dataloader)
        self._next_batch_id = 0

        self._httpd: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("server not started")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if self._httpd is not None:
            raise RuntimeError("server already started")

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

            def _write_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _query(self) -> dict[str, str]:
                path, _, query_str = self.path.partition("?")
                q: dict[str, str] = {}
                if query_str:
                    for token in query_str.split("&"):
                        if not token:
                            continue
                        k, _, v = token.partition("=")
                        q[k] = v
                self.path = path
                return q

            def do_GET(self) -> None:  # noqa: N802
                try:
                    q = self._query()
                    if self.path == "/current_batch_id":
                        local_rank = int(q["local_rank"])
                        batch_id = owner.current_batch_id(local_rank)
                        self._write_json(200, {"batch_id": batch_id})
                        return
                    self._write_json(404, {"error": "not found"})
                except Exception as exc:  # noqa: BLE001
                    self._write_json(400, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    q = self._query()
                    if self.path == "/seek":
                        batch_id = int(q["batch_id"])
                        print(f"Received seek request for batch_id={batch_id}")
                        owner.seek(batch_id)
                        print(f"Completed seek request for batch_id={batch_id}")
                        self._write_json(200, {"batch_id": batch_id})
                        return
                    if self.path == "/fetch_next":
                        local_rank = int(q["local_rank"])
                        print(f"Received fetch_next request for local_rank={local_rank}")
                        filename = owner.fetch_next(local_rank)
                        print(f"Returning fetch_next response for local_rank={local_rank}, filename={filename}")
                        self._write_json(200, {"filename": filename})
                        return
                    self._write_json(404, {"error": "not found"})
                except _NotReadyError as exc:
                    self._write_json(409, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    self._write_json(400, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._http_thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=5)
        self._httpd = None
        self._http_thread = None

    def seek(self, batch_id: int) -> None:
        if batch_id < 0:
            raise ValueError("batch_id must be >= 0")
        with self._lock:
            self._state.window_start_batch_id = batch_id
            self._state.fetched_current_by_rank = [False for _ in range(self.local_rank_count)]
            self._state.current_batch_by_rank = [batch_id - 1 for _ in range(self.local_rank_count)]
            for rank in range(self.local_rank_count):
                self._clear_rank_files(rank)
            
            # Create new dataloader starting at the seek position
            self._dataloader = self.dataloader_start_at_batch_id(batch_id)
            self._dataloader_iter = iter(self._dataloader)
            self._next_batch_id = batch_id
            
            self._prefetch_window_locked()

    def current_batch_id(self, local_rank: int) -> int:
        self._validate_rank(local_rank)
        with self._lock:
            return self._state.current_batch_by_rank[local_rank]

    def fetch_next(self, local_rank: int) -> str:
        self._validate_rank(local_rank)
        with self._lock:
            self._prefetch_window_locked()
            batch_id = self._state.window_start_batch_id
            if self._state.fetched_current_by_rank[local_rank]:
                raise _NotReadyError(f"local_rank={local_rank} already fetched batch_id={batch_id}; waiting other ranks")

            filename = self._rank_file(local_rank, batch_id)
            if not filename.exists():
                self._ensure_rank_link_locked(local_rank, batch_id)

            self._state.current_batch_by_rank[local_rank] = batch_id
            self._state.fetched_current_by_rank[local_rank] = True

            if all(self._state.fetched_current_by_rank):
                self._state.window_start_batch_id += 1
                self._state.fetched_current_by_rank = [False for _ in range(self.local_rank_count)]
                self._prefetch_window_locked()

            return str(filename)

    def _validate_rank(self, local_rank: int) -> None:
        if local_rank < 0 or local_rank >= self.local_rank_count:
            raise ValueError(f"invalid local_rank: {local_rank}")

    def _clear_rank_files(self, local_rank: int) -> None:
        rank_dir = self.rank_dirs[local_rank]
        for p in rank_dir.glob("*.tbf"):
            p.unlink(missing_ok=True)

    def _shared_file(self, batch_id: int, ap_rank: int, ring_attn_rank: int) -> Path:
        return self.shared_dir / f"batch_{batch_id}_ap{ap_rank}_ring{ring_attn_rank}.tbf"

    def _rank_file(self, local_rank: int, batch_id: int) -> Path:
        return self.rank_dirs[local_rank] / f"batch_{batch_id}.tbf"

    def _materialize_shared_locked(self, batch_id: int) -> None:
        # Check if already materialized
        if self._shared_file(batch_id, 0, 0).exists():
            return

        # Ensure we're reading batches sequentially
        if batch_id != self._next_batch_id:
            raise RuntimeError(
                f"batch_id mismatch: expected {self._next_batch_id}, got {batch_id}. "
                "Batches must be materialized sequentially."
            )

        try:
            global_batch = next(self._dataloader_iter)
        except StopIteration as exc:
            raise ValueError(f"no data for batch_id={batch_id}") from exc

        self._next_batch_id += 1

        for ap_rank, funcs_row in enumerate(self.to_records_funcs):
            for ring_attn_rank, to_records_func in enumerate(funcs_row):
                shared = self._shared_file(batch_id, ap_rank, ring_attn_rank)
                records = to_records_func(global_batch)
                if not isinstance(records, list):
                    raise TypeError("to_records must return list[dict[str, Tensor]]")
                tmp_path = shared.with_suffix(".tmp")
                write_tbf(tmp_path, records, page_size=self.page_size)
                os.replace(tmp_path, shared)

    def _ensure_rank_link_locked(self, local_rank: int, batch_id: int) -> Path:
        target = self._rank_file(local_rank, batch_id)
        if target.exists():
            return target
        self._materialize_shared_locked(batch_id)
        ap_rank = self.rank_ap_mapping[local_rank]
        ring_attn_rank = self.rank_ring_attn_mapping[local_rank]
        shared = self._shared_file(batch_id, ap_rank, ring_attn_rank)
        try:
            os.link(shared, target)
        except FileExistsError:
            pass
        return target

    def _prefetch_window_locked(self) -> None:
        start = self._state.window_start_batch_id
        end = start + self.prefetch_count
        for batch_id in range(start, end):
            for rank in range(self.local_rank_count):
                if batch_id == start and self._state.fetched_current_by_rank[rank]:
                    continue
                self._ensure_rank_link_locked(rank, batch_id)
