"""
Verification Script for Distributed Batch Predictions.
Reads the prediction outputs directly from HDFS via Apache Spark or local Parquet directory
and verifies schema, record count (1,000,000), partitions, evaluation metrics, and zone metrics.
"""

import os
import sys
import time
import pandas as pd
import pyarrow.parquet as pq


def verify_predictions_spark(base_hdfs_path: str = "hdfs://namenode:9000/rental_data/predictions"):
    from pyspark.sql import SparkSession

    print("\n" + "=" * 80)
    print(" VERIFYING DISTRIBUTED BATCH PREDICTIONS IN HDFS VIA SPARK")
    print(f" Predictions Base HDFS URI: {base_hdfs_path}")
    print("=" * 80)

    start_time = time.time()

    spark = (
        SparkSession.builder.appName("VerifyBatchPredictions")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    batch_pred_path = f"{base_hdfs_path}/batch_predictions"
    print(f"\n[1/3] Reading Batch Predictions from: {batch_pred_path}")
    pred_df = spark.read.parquet(batch_pred_path)
    total_preds = pred_df.count()
    partitions = pred_df.rdd.getNumPartitions()

    print(f"  - Record Count:         {total_preds:,} (Expected: 1,000,000)")
    print(f"  - Partition Count:      {partitions}")
    print("  - Schema:")
    pred_df.printSchema()

    assert total_preds == 1_000_000, f"Expected 1,000,000 predictions, but found {total_preds}!"

    print("\n  - Sample 5 Predictions:")
    pred_df.show(5, truncate=False)

    eval_metrics_path = f"{base_hdfs_path}/evaluation_metrics"
    print(f"\n[2/3] Reading Evaluation Metrics from: {eval_metrics_path}")
    metrics_df = spark.read.parquet(eval_metrics_path)
    metrics_df.show(truncate=False)

    zone_metrics_path = f"{base_hdfs_path}/zone_metrics"
    print(f"\n[3/3] Reading Zone Metrics from: {zone_metrics_path}")
    zone_df = spark.read.parquet(zone_metrics_path)
    zone_df.show(truncate=False)

    elapsed = time.time() - start_time
    print(f"\nAll Batch Prediction Outputs Successfully Verified in HDFS in {elapsed:.2f} seconds.")
    print("=" * 80 + "\n")

    spark.stop()
    return {
        "status": "SUCCESS",
        "total_predictions": total_preds,
        "elapsed_seconds": elapsed,
    }


def verify_predictions_standalone(base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bigdata", "predictions")
    base_dir = os.path.abspath(base_dir)

    print("\n" + "=" * 80)
    print(" VERIFYING BATCH PREDICTIONS (STANDALONE PARQUET VERIFICATION)")
    print(f" Predictions Directory: {base_dir}")
    print("=" * 80)

    start_time = time.time()

    # 1. Verify Batch Predictions
    preds_dir = os.path.join(base_dir, "batch_predictions")
    if not os.path.exists(preds_dir):
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")

    pred_files = sorted([os.path.join(preds_dir, f) for f in os.listdir(preds_dir) if f.endswith(".parquet")])
    total_preds = 0
    sample_rows = None

    for f in pred_files:
        table = pq.read_table(f)
        total_preds += len(table)
        if sample_rows is None:
            sample_rows = table.slice(0, 5).to_pandas()

    print(f"\n[1/3] Parquet Prediction Partitions: {len(pred_files)}")
    print(f"  - Total Predictions Count: {total_preds:,}")
    print(f"  - Schema: {[f.name for f in table.schema]}")
    print("\n  - Sample 5 Predictions:")
    print(sample_rows.to_string(index=False))

    assert total_preds == 1_000_000, f"Expected 1,000,000 predictions, but found {total_preds}!"

    # 2. Verify Evaluation Metrics
    eval_file = os.path.join(base_dir, "evaluation_metrics", "part-00000.parquet")
    eval_df = pq.read_table(eval_file).to_pandas()
    print("\n[2/3] Global Evaluation Metrics Table:")
    print(eval_df.to_string(index=False))

    # 3. Verify Zone Metrics
    zone_file = os.path.join(base_dir, "zone_metrics", "part-00000.parquet")
    zone_df = pq.read_table(zone_file).to_pandas()
    print("\n[3/3] Zone-Level Evaluation Metrics Table:")
    print(zone_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nAll Batch Prediction Outputs Successfully Verified in {elapsed:.2f} seconds.")
    print("=" * 80 + "\n")
    return {
        "status": "SUCCESS",
        "total_predictions": total_preds,
        "elapsed_seconds": elapsed,
    }


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target.startswith("hdfs://"):
        verify_predictions_spark(target)
    else:
        verify_predictions_standalone(target)


if __name__ == "__main__":
    main()
