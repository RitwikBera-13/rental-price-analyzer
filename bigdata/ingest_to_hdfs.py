"""
HDFS Ingestion Module for Rental Price Analyzer.
Uploads the generated partitioned Parquet dataset into Hadoop HDFS under:
/rental_data/raw/mumbai_rentals_large/
"""

import os
import sys
import time
import logging
from typing import Dict, Any

from .config import get_config
from .hdfs_client import get_hdfs_client

logger = logging.getLogger("HDFSIngestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def ingest_to_hdfs(
    local_dir: str = None,
    hdfs_target_dir: str = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Upload all Parquet partitions from local directory into HDFS.
    """
    cfg = get_config()
    local_dir = local_dir or cfg.LOCAL_LARGE_DATASET_DIR
    hdfs_target_dir = hdfs_target_dir or cfg.HDFS_LARGE_RAW_DIR

    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Local Parquet directory not found: {local_dir}. Run generate_large_dataset.py first.")

    parquet_files = sorted([f for f in os.listdir(local_dir) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {local_dir}")

    total_local_bytes = sum(os.path.getsize(os.path.join(local_dir, f)) for f in parquet_files)
    total_local_mb = total_local_bytes / (1024 * 1024)

    client = get_hdfs_client()
    print(f"\n[1/3] Connecting to HDFS...")
    print(f"  Connection mode: {'CONNECTED (HDFS Cluster)' if client.is_connected else 'LOCAL SIMULATION (Fallback)'}")
    print(f"  Target HDFS path: {hdfs_target_dir}")
    print(f"  Local source:     {local_dir} ({len(parquet_files)} files, {total_local_mb:.2f} MB)")

    # Ensure target directory exists on HDFS
    client.mkdir(hdfs_target_dir)

    print(f"\n[2/3] Ingesting {len(parquet_files)} Parquet partition files to HDFS...")
    start_time = time.time()
    uploaded = {}

    for i, fname in enumerate(parquet_files, start=1):
        loc_path = os.path.join(local_dir, fname)
        target_path = f"{hdfs_target_dir}/{fname}"
        f_size_mb = os.path.getsize(loc_path) / (1024 * 1024)

        t0 = time.time()
        ok = client.upload_file(loc_path, target_path, overwrite=overwrite)
        t_el = time.time() - t0
        uploaded[fname] = ok

        status_str = "SUCCESS" if ok else "FAILED"
        print(f"  [{i:02d}/{len(parquet_files):02d}] {fname} ({f_size_mb:.2f} MB) -> {target_path} [{status_str} in {t_el:.2f}s]")

    elapsed = time.time() - start_time
    print(f"\n[3/3] Verifying HDFS storage...")
    hdfs_items = client.list_dir(hdfs_target_dir)
    total_hdfs_bytes = sum(item.get("length", 0) for item in hdfs_items)
    total_hdfs_mb = total_hdfs_bytes / (1024 * 1024)

    print(f"  - Files in HDFS directory: {len(hdfs_items)}")
    print(f"  - Total HDFS data size:    {total_hdfs_mb:.2f} MB ({total_hdfs_bytes:,} bytes)")
    print(f"  - Total ingestion time:    {elapsed:.2f} seconds")

    return {
        "files_uploaded": len(uploaded),
        "successful_uploads": sum(1 for v in uploaded.values() if v),
        "hdfs_target_dir": hdfs_target_dir,
        "hdfs_items": hdfs_items,
        "total_hdfs_bytes": total_hdfs_bytes,
        "total_hdfs_mb": total_hdfs_mb,
    }


if __name__ == "__main__":
    ingest_to_hdfs()
