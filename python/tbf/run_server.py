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
    parser = argparse.ArgumentParser(description="Start TBFBatchHTTPServer")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--prefetch-count", type=int, default=2, help="Number of batches to prefetch (default: 2)")
    parser.add_argument("--local-rank-count", type=int, default=1, help="Number of local ranks (default: 1)")
    parser.add_argument("--local-dir", required=True, help="Local directory for storing batch files")
    parser.add_argument("--page-size", type=int, default=4096, help="TBF page size (default: 4096)")
    
    args = parser.parse_args()
    
    # Create server
    server = TBFBatchHTTPServer(
        dataloader_start_at_batch_id=dataloader_start_at_batch_id,
        to_records=to_records,
        prefetch_count=args.prefetch_count,
        local_rank_count=args.local_rank_count,
        local_dir=args.local_dir,
        page_size=args.page_size,
    )
    
    # Start server
    server.start(host=args.host, port=args.port)
    print(f"Server started at {server.base_url}")
    print(f"Local directory: {args.local_dir}")
    print(f"Prefetch count: {args.prefetch_count}")
    print(f"Local rank count: {args.local_rank_count}")
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
