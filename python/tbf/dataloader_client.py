from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .reader import TBFReader


class AsyncTBFBatchClient:
    def __init__(
        self,
        base_url: str,
        local_rank: int,
        queue_size: int = 2,
        poll_interval_sec: float = 0.01,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.local_rank = local_rank
        self.poll_interval_sec = poll_interval_sec

        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def seek(self, batch_id: int) -> None:
        self._post_json(f"/seek?batch_id={batch_id}")

    def current_batch_id(self) -> int:
        payload = self._get_json(f"/current_batch_id?local_rank={self.local_rank}")
        return int(payload["batch_id"])

    def fetch_next_filename(self) -> str:
        payload = self._post_json(f"/fetch_next?local_rank={self.local_rank}")
        return str(payload["filename"])

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    def batches(self):
        self.start()
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return
            try:
                batch = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            yield batch

    def _worker(self) -> None:
        pending_batch: list[dict[str, Any]] | None = None
        while not self._stop_event.is_set():
            try:
                if pending_batch is None:
                    filename = self.fetch_next_filename()
                    pending_batch = self._load_records(filename)
                self._queue.put(pending_batch, timeout=self.poll_interval_sec)
                pending_batch = None
            except queue.Full:
                continue
            except RuntimeError:
                time.sleep(self.poll_interval_sec)

    def _load_records(self, filename: str):
        with TBFReader(filename) as reader:
            os.unlink(filename)
            out = []
            for i in range(len(reader)):
                out.append(reader[i])
            return out

    def _get_json(self, path_with_query: str) -> dict[str, object]:
        return self._request_json("GET", path_with_query)

    def _post_json(self, path_with_query: str) -> dict[str, object]:
        return self._request_json("POST", path_with_query)

    def _request_json(self, method: str, path_with_query: str) -> dict[str, object]:
        req = urllib.request.Request(
            url=f"{self.base_url}{path_with_query}",
            method=method,
            data=b"" if method == "POST" else None,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                body = resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid response: {payload}")
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        return payload
