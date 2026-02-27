#!/usr/bin/env python3
"""Simple CLI tool to start TBFBatchHTTPServer with SFT dataloader."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloders.sft import dataloader_start_at_batch_id, to_records
from tbf.dataloader_server import TBFBatchHTTPServer


def main():
    prefetch_count = 2
    local_rank_count = 8
    local_dir = "./tbf_local"
    page_size = 4096
    host = "127.0.0.1"
    port = 8999
    
    # Create server
    server = TBFBatchHTTPServer(
        dataloader_start_at_batch_id=dataloader_start_at_batch_id,
        to_records_funcs=[[to_records]],
        rank_ap_mapping=[0] * local_rank_count,
        rank_ring_attn_mapping=[0] * local_rank_count,
        prefetch_count=prefetch_count,
        local_rank_count=local_rank_count,
        local_dir=local_dir,
        page_size=page_size,
    )
    
    # Start server
    server.start(host=host, port=port)
    print(f"Server started at {server.base_url}")
    print(f"Local directory: {local_dir}")
    print(f"Prefetch count: {prefetch_count}")
    print(f"Local rank count: {local_rank_count}")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
        print("Server stopped.")


if __name__ == "__main__":
    main()
