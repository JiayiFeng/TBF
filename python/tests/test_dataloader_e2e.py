from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tbf.dataloader_client import AsyncTBFBatchClient
from tbf.dataloader_server import Link, Own, TBFBatchHTTPServer


class MNISTLikeDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.images = torch.arange(size * 28 * 28, dtype=torch.float32).reshape(size, 1, 28, 28)
        self.labels = torch.arange(size, dtype=torch.int64) % 10

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def test_tbf_dataloader_server_client_e2e(tmp_path: Path) -> None:
    dataset = MNISTLikeDataset(size=32)
    global_batch_size = 8
    micro_batch_size = 2

    def dataloader_for_batch_id(batch_id: int) -> DataLoader:
        start = (batch_id * global_batch_size) % len(dataset)
        indices = [(start + i) % len(dataset) for i in range(global_batch_size)]
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=global_batch_size, shuffle=False, num_workers=0)

    def to_records(global_batch: object) -> list[Own | Link]:
        assert isinstance(global_batch, (tuple, list))
        images, labels = global_batch
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        records_rank0: list[dict[str, torch.Tensor]] = []
        for start in range(0, int(images.shape[0]), micro_batch_size):
            end = start + micro_batch_size
            records_rank0.append({
                "image": images[start:end].clone(),
                "label": labels[start:end].clone(),
            })
        return [Own(records_rank0), Link(src_rank=0)]

    server = TBFBatchHTTPServer(
        dataset=dataset,
        dataloader_for_batch_id=dataloader_for_batch_id,
        to_records=to_records,
        prefetch_count=2,
        local_rank_count=2,
        local_dir=tmp_path,
    )
    server.start(host="127.0.0.1", port=0)

    client0 = AsyncTBFBatchClient(base_url=server.base_url, local_rank=0, queue_size=2)
    client1 = AsyncTBFBatchClient(base_url=server.base_url, local_rank=1, queue_size=2)

    try:
        client0.seek(0)
        assert client0.current_batch_id() == -1
        assert client1.current_batch_id() == -1

        rank0_batch0 = tmp_path / "rank_0" / "batch_0.tbf"
        rank1_batch0 = tmp_path / "rank_1" / "batch_0.tbf"
        assert rank0_batch0.exists()
        assert rank1_batch0.exists()
        s0 = os.stat(rank0_batch0)
        s1 = os.stat(rank1_batch0)
        assert s0.st_ino == s1.st_ino

        it = client0.batches()
        batch0 = next(it)
        assert len(batch0) == global_batch_size // micro_batch_size

        for micro in batch0:
            assert set(micro.keys()) == {"image", "label"}
            assert tuple(micro["image"].shape) == (micro_batch_size, 1, 28, 28)
            assert tuple(micro["label"].shape) == (micro_batch_size,)

        batch0_rank1_path = client1.fetch_next_filename()
        batch0_rank1 = client1._load_records(batch0_rank1_path)
        assert len(batch0_rank1) == len(batch0)
        for micro0, micro1 in zip(batch0, batch0_rank1):
            assert torch.equal(micro0["image"], micro1["image"])
            assert torch.equal(micro0["label"], micro1["label"])

        # Both files are consumed and unlinked after opening.
        assert not rank0_batch0.exists()
        assert not rank1_batch0.exists()

        assert client0.current_batch_id() >= 0
        assert client1.current_batch_id() >= 0

        # Barrier passed after both ranks fetched batch0; rank0 can move to batch1.

        batch1 = next(it)
        assert len(batch1) == global_batch_size // micro_batch_size
        assert client0.current_batch_id() >= 1
    finally:
        client0.stop()
        client1.stop()
        server.stop()


def test_tbf_dataloader_client_queue_full_does_not_drop_batches(tmp_path: Path) -> None:
    dataset = MNISTLikeDataset(size=128)
    global_batch_size = 4
    micro_batch_size = 2

    def dataloader_for_batch_id(batch_id: int) -> DataLoader:
        start = (batch_id * global_batch_size) % len(dataset)
        indices = [(start + i) % len(dataset) for i in range(global_batch_size)]
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=global_batch_size, shuffle=False, num_workers=0)

    def to_records(global_batch: object) -> list[Own]:
        assert isinstance(global_batch, (tuple, list))
        images, labels = global_batch
        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)

        records: list[dict[str, torch.Tensor]] = []
        for start in range(0, int(images.shape[0]), micro_batch_size):
            end = start + micro_batch_size
            records.append({
                "image": images[start:end].clone(),
                "label": labels[start:end].clone(),
            })
        return [Own(records)]

    server = TBFBatchHTTPServer(
        dataset=dataset,
        dataloader_for_batch_id=dataloader_for_batch_id,
        to_records=to_records,
        prefetch_count=8,
        local_rank_count=1,
        local_dir=tmp_path,
    )
    server.start(host="127.0.0.1", port=0)

    client = AsyncTBFBatchClient(
        base_url=server.base_url,
        local_rank=0,
        queue_size=1,
        poll_interval_sec=0.01,
    )

    try:
        client.seek(0)
        client.start()
        time.sleep(0.2)

        # With a full queue, worker should not keep fetching and skipping batches.
        assert client.current_batch_id() <= 1

        it = client.batches()
        batch0 = next(it)
        batch1 = next(it)

        assert int(batch0[0]["label"][0]) == 0
        assert int(batch1[0]["label"][0]) == global_batch_size
    finally:
        client.stop()
        server.stop()
