from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from tbf.dataloader_client import AsyncTBFBatchClient
from tbf.dataloader_server import TBFBatchHTTPServer


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

    def dataloader_start_at_batch_id(batch_id: int) -> DataLoader:
        start_index = (batch_id * global_batch_size) % len(dataset)
        # Create indices for all samples from start_index onwards (wrapping around)
        indices = [(start_index + i) % len(dataset) for i in range(len(dataset))]
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=global_batch_size, shuffle=False, num_workers=0)

    def to_records(global_batch: object) -> list[dict[str, torch.Tensor]]:
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
        return records

    server = TBFBatchHTTPServer(
        dataloader_start_at_batch_id=dataloader_start_at_batch_id,
        to_records_funcs=[[to_records]],
        rank_ap_mapping=[0, 0],
        rank_ring_attn_mapping=[0, 0],
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

        # Client rank0 deletes file after opening.
        assert not rank0_batch0.exists()
        # rank1 has not fetched yet, so its file still exists.
        assert rank1_batch0.exists()

        assert client0.current_batch_id() == 0
        assert client1.current_batch_id() == -1

        # Barrier: once rank1 fetches batch0, window can advance and rank0 gets batch1.
        client1.fetch_next_filename()
        assert client1.current_batch_id() == 0

        batch1 = next(it)
        assert len(batch1) == global_batch_size // micro_batch_size
        assert client0.current_batch_id() >= 1
    finally:
        client0.stop()
        client1.stop()
        server.stop()
