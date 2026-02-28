from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, TypeAlias

from .format import DEFAULT_PAGE_SIZE
from .writer import write_tbf


class _NotReadyError(RuntimeError):
    pass


Record: TypeAlias = dict[str, Any]
Records: TypeAlias = list[Record]


@dataclass(frozen=True)
class Own:
    records: Records


@dataclass(frozen=True)
class Link:
    src_rank: int


RankOutput: TypeAlias = Own | Link
ToRecordsResult: TypeAlias = list[RankOutput]


@dataclass
class _ServerState:
    window_start_batch_id: int
    fetched_current_by_rank: list[bool]
    current_batch_by_rank: list[int]


class TBFBatchHTTPServer:
    def __init__(
        self,
        dataset: Any,
        dataloader_for_batch_id: Callable[[int], Any],
        to_records: Callable[[Any], ToRecordsResult],
        prefetch_count: int,
        local_rank_count: int,
        local_dir: str | Path,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        if prefetch_count <= 0:
            raise ValueError("prefetch_count must be > 0")
        if local_rank_count <= 0:
            raise ValueError("local_rank_count must be > 0")

        self.dataset = dataset
        self.dataloader_for_batch_id = dataloader_for_batch_id
        self.to_records = to_records
        self.prefetch_count = prefetch_count
        self.local_rank_count = local_rank_count
        self.page_size = page_size

        self.local_dir = Path(local_dir)
        self.rank_dirs = [self.local_dir / f"rank_{r}" for r in range(local_rank_count)]
        for rank_dir in self.rank_dirs:
            rank_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._state = _ServerState(
            window_start_batch_id=0,
            fetched_current_by_rank=[False for _ in range(local_rank_count)],
            current_batch_by_rank=[-1 for _ in range(local_rank_count)],
        )

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
                        owner.seek(batch_id)
                        self._write_json(200, {"batch_id": batch_id})
                        return
                    if self.path == "/fetch_next":
                        local_rank = int(q["local_rank"])
                        filename = owner.fetch_next(local_rank)
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

    def _rank_file(self, local_rank: int, batch_id: int) -> Path:
        return self.rank_dirs[local_rank] / f"batch_{batch_id}.tbf"

    def _to_records_result(self, batch_id: int) -> ToRecordsResult:
        loader = self.dataloader_for_batch_id(batch_id)
        iterator = iter(loader)
        try:
            global_batch = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"no data for batch_id={batch_id}") from exc

        outputs = self.to_records(global_batch)
        if not isinstance(outputs, list):
            raise TypeError("to_records must return list[Own | Link]")
        if len(outputs) != self.local_rank_count:
            raise ValueError(
                f"to_records returned {len(outputs)} rank outputs, expected {self.local_rank_count}"
            )
        has_owner = False
        for rank, output in enumerate(outputs):
            if isinstance(output, Own):
                has_owner = True
                continue
            if not isinstance(output, Link):
                raise TypeError(f"invalid rank output for rank={rank}: {type(output)!r}")
            src_rank = output.src_rank
            if src_rank < 0 or src_rank >= self.local_rank_count:
                raise ValueError(f"invalid src_rank={src_rank} for rank={rank}")
            if src_rank == rank:
                raise ValueError(f"rank={rank} cannot link to itself")
            src_output = outputs[src_rank]
            if not isinstance(src_output, Own):
                raise ValueError(f"rank={rank} links to rank={src_rank}, which is not Own")

        if not has_owner:
            raise ValueError("to_records result must include at least one Own output")
        return outputs

    def _materialize_batch_locked(self, batch_id: int, skip_ranks: set[int] | None = None) -> None:
        skip = skip_ranks or set()
        outputs = self._to_records_result(batch_id)
        for rank, output in enumerate(outputs):
            if rank in skip:
                continue
            if not isinstance(output, Own):
                continue
            target = self._rank_file(rank, batch_id)
            if target.exists():
                continue
            tmp_path = target.with_suffix(".tmp")
            write_tbf(tmp_path, output.records, page_size=self.page_size)
            os.replace(tmp_path, target)

        for rank, output in enumerate(outputs):
            if rank in skip:
                continue
            if not isinstance(output, Link):
                continue
            target = self._rank_file(rank, batch_id)
            if target.exists():
                continue
            source = self._rank_file(output.src_rank, batch_id)
            try:
                os.link(source, target)
            except FileExistsError:
                pass

    def _prefetch_window_locked(self) -> None:
        start = self._state.window_start_batch_id
        end = start + self.prefetch_count
        for batch_id in range(start, end):
            if batch_id == start:
                skipped = {rank for rank, fetched in enumerate(self._state.fetched_current_by_rank) if fetched}
                self._materialize_batch_locked(batch_id, skip_ranks=skipped)
                continue
            self._materialize_batch_locked(batch_id)
