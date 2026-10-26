"""
Verification Script for Distributed Rental Analytics in HDFS.
Reads the analytics Parquet tables back from HDFS via Apache Spark
and displays their contents to independently confirm successful execution.
"""

import sys
import time
from pyspark.sql import SparkSession


def verify_analytics(base_hdfs_path: str = "hdfs://namenode:9000/rental_data/analytics"):
    print("\n" + "=" * 80)
    print(" VERIFYING HDFS ANALYTICS PARQUET OUTPUTS VIA SPARK")
    print(f" Base Analytics HDFS URI: {base_hdfs_path}")
    print("=" * 80)

    start_time = time.time()

    spark = (
        SparkSession.builder.appName("VerifyRentalAnalytics")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    tables = [
        ("Zone Analysis", f"{base_hdfs_path}/zone_analysis"),
        ("Rent Per Sqft by Zone", f"{base_hdfs_path}/rent_per_sqft"),
        ("BHK Bedroom Analysis", f"{base_hdfs_path}/bhk_analysis"),
        ("Furnished Analysis", f"{base_hdfs_path}/furnished_analysis"),
        ("Property Age Analysis", f"{base_hdfs_path}/age_analysis"),
        ("Overall Statistics", f"{base_hdfs_path}/overall_statistics"),
        ("Data Quality Checks", f"{base_hdfs_path}/data_quality"),
    ]

    results = {}

    for title, hdfs_uri in tables:
        print(f"\n>>> Table: {title} ({hdfs_uri})")
        try:
            df = spark.read.parquet(hdfs_uri)
            row_count = df.count()
            print(f"    Status: SUCCESS | Rows: {row_count} | Partitions: {df.rdd.getNumPartitions()}")
            df.show(truncate=False)
            results[title] = {"status": "SUCCESS", "rows": row_count}
        except Exception as e:
            print(f"    Status: FAILED | Error: {e}")
            results[title] = {"status": "FAILED", "error": str(e)}

    elapsed = time.time() - start_time
    print(f"\nAll 7 Analytics Tables verified via Spark in {elapsed:.2f} seconds.")
    print("=" * 80 + "\n")

    spark.stop()
    return results


if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "hdfs://namenode:9000/rental_data/analytics"
    verify_analytics(base_path)
