#!/usr/bin/env python3
"""启动 TBF SFT 数据加载服务器。

这个脚本使用 mmq.data.sft_dataloader 中的 tbf_dataloader_start_at_batch_id 和 
tbf_to_records 函数来启动 TBF HTTP 服务器。
"""

import argparse
import os
import sys
import time
from tbf.dataloader_server import TBFBatchHTTPServer
from mmq.data.sft_tbf_dataloader import tbf_dataloader_start_at_batch_id, tbf_to_records


def main():
    host = "127.0.0.1"
    port = 8999
    prefetch_count = 2
    local_rank_count = 8
    local_dir = "/tbf_local"
    page_size = 4096

    
    print("=" * 70)
    print("TBF SFT 数据加载服务器")
    print("=" * 70)
    print(f"配置:")
    print(f"  Host:              {host}")
    print(f"  Port:              {port}")
    print(f"  Prefetch Count:    {prefetch_count}")
    print(f"  Local Rank Count:  {local_rank_count}")
    print(f"  Local Directory:   {local_dir}")
    print(f"  Page Size:         {page_size} bytes")
    print("=" * 70)
    
    # 创建服务器
    server = TBFBatchHTTPServer(
        dataloader_start_at_batch_id=tbf_dataloader_start_at_batch_id,
        to_records=tbf_to_records,
        prefetch_count=prefetch_count,
        local_rank_count=local_rank_count,
        local_dir=local_dir,
        page_size=page_size,
    )
    
    # 启动服务器
    server.start(host=host, port=port)
    
    print(f"\n✓ 服务器已启动: {server.base_url}")
    print(f"\n使用示例:")
    print(f"  - 获取当前批次ID: curl '{server.base_url}/current_batch_id?local_rank=0'")
    print(f"  - 跳转到批次:     curl -X POST '{server.base_url}/seek?batch_id=100'")
    print(f"  - 获取下一批次:   curl -X POST '{server.base_url}/fetch_next?local_rank=0'")
    print(f"\n按 Ctrl+C 停止服务器...")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在停止服务器...")
        server.stop()
        print("✓ 服务器已停止")


if __name__ == "__main__":
    main()
