"""
Distributed Rental Analytics Pipeline using Apache Spark / PySpark.
Performs distributed aggregations on 1,000,000 rental records stored in HDFS
and saves analytical summary datasets back to HDFS in Parquet format.
"""

import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def run_rental_analytics(
    input_hdfs_path: str = "hdfs://namenode:9000/rental_data/raw/mumbai_rentals_large",
    analytics_base_hdfs: str = "hdfs://namenode:9000/rental_data/analytics",
):
    print("\n" + "=" * 80)
    print(" DISTRIBUTED RENTAL ANALYTICS - APACHE SPARK")
    print(f" Input HDFS Path:      {input_hdfs_path}")
    print(f" Analytics Output Path: {analytics_base_hdfs}")
    print("=" * 80)

    start_time = time.time()

    # 1. Initialize SparkSession configured for HDFS cluster execution
    # [Distributed Context Initialization]
    spark = (
        SparkSession.builder.appName("DistributedRentalAnalytics")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print(f"\n[1/8] Spark Environment:")
    print(f"  - Spark Version:     {spark.version}")
    print(f"  - Spark Master:      {spark.conf.get('spark.master', 'local[*]')}")
    print(f"  - Default FS:        hdfs://namenode:9000")

    # 2. Read dataset directly from HDFS into a distributed Spark DataFrame
    # [Distributed Read from HDFS DataNodes]
    t0 = time.time()
    raw_df = spark.read.parquet(input_hdfs_path)
    total_records = raw_df.count()
    num_partitions = raw_df.rdd.getNumPartitions()
    print(f"\n[2/8] Loaded HDFS Dataset in {time.time() - t0:.2f}s:")
    print(f"  - Record Count:      {total_records:,}")
    print(f"  - DataFrame Partitions: {num_partitions}")
    print(f"  - Schema:")
    raw_df.printSchema()

    # Cache/persist DataFrame in memory for optimal multi-pass query execution
    # [Distributed In-Memory Caching across Executors]
    raw_df = raw_df.cache()

    # Add calculated columns in a distributed transformation (lazy evaluation)
    # [Distributed Transformations]
    enriched_df = raw_df.withColumn(
        "rent_per_sqft",
        F.round(F.col("rent") / F.col("area_sqft"), 2)
    ).withColumn(
        "furnished_label",
        F.when(F.col("furnished") == 1, "Furnished").otherwise("Unfurnished")
    ).withColumn(
        "age_group",
        F.when(F.col("age_years") <= 5, "0-5 years")
        .when(F.col("age_years") <= 10, "6-10 years")
        .when(F.col("age_years") <= 20, "11-20 years")
        .when(F.col("age_years") <= 30, "21-30 years")
        .otherwise("31+ years")
    )
    enriched_df = enriched_df.cache()

    # --------------------------------------------------------------------------
    # A. Zone-level analysis
    # [Distributed GroupBy & Aggregations across Partitions]
    # --------------------------------------------------------------------------
    print("\n[A] Zone-Level Analysis:")
    zone_analysis_df = (
        enriched_df.groupBy("zone")
        .agg(
            F.count("*").alias("property_count"),
            F.round(F.avg("area_sqft"), 1).alias("avg_area_sqft"),
            F.round(F.avg("rent"), 2).alias("avg_rent"),
            F.round(F.min("rent"), 2).alias("min_rent"),
            F.round(F.max("rent"), 2).alias("max_rent"),
        )
        .orderBy(F.desc("property_count"))
    )
    zone_analysis_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # B. Rent per square foot analysis by zone
    # [Distributed PSF Statistics]
    # --------------------------------------------------------------------------
    print("\n[B] Rent Per Square Foot (PSF) Analysis by Zone:")
    rent_psf_df = (
        enriched_df.groupBy("zone")
        .agg(
            F.count("*").alias("property_count"),
            F.round(F.avg("rent_per_sqft"), 2).alias("avg_rent_per_sqft"),
            F.round(F.min("rent_per_sqft"), 2).alias("min_rent_per_sqft"),
            F.round(F.max("rent_per_sqft"), 2).alias("max_rent_per_sqft"),
        )
        .orderBy(F.desc("avg_rent_per_sqft"))
    )
    rent_psf_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # C. BHK analysis (Bedroom distribution)
    # [Distributed Grouping on Bedrooms]
    # --------------------------------------------------------------------------
    print("\n[C] BHK (Bedroom Count) Analysis:")
    bhk_analysis_df = (
        enriched_df.groupBy("bedrooms")
        .agg(
            F.count("*").alias("property_count"),
            F.round(F.avg("rent"), 2).alias("avg_rent"),
            F.round(F.avg("area_sqft"), 1).alias("avg_area_sqft"),
            F.round(F.avg("rent_per_sqft"), 2).alias("avg_rent_per_sqft"),
        )
        .orderBy("bedrooms")
    )
    bhk_analysis_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # D. Furnished vs Unfurnished analysis
    # [Distributed Categorical Comparison]
    # --------------------------------------------------------------------------
    print("\n[D] Furnished vs Unfurnished Analysis:")
    furnished_analysis_df = (
        enriched_df.groupBy("furnished", "furnished_label")
        .agg(
            F.count("*").alias("property_count"),
            F.round(F.avg("rent"), 2).alias("avg_rent"),
            F.round(F.avg("area_sqft"), 1).alias("avg_area_sqft"),
            F.round(F.avg("rent_per_sqft"), 2).alias("avg_rent_per_sqft"),
        )
        .orderBy("furnished")
    )
    furnished_analysis_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # E. Property age analysis (Binned Age Brackets)
    # [Distributed Bracket Aggregations]
    # --------------------------------------------------------------------------
    print("\n[E] Property Age Group Analysis:")
    age_analysis_df = (
        enriched_df.groupBy("age_group")
        .agg(
            F.count("*").alias("property_count"),
            F.round(F.avg("rent"), 2).alias("avg_rent"),
            F.round(F.avg("area_sqft"), 1).alias("avg_area_sqft"),
            F.round(F.avg("rent_per_sqft"), 2).alias("avg_rent_per_sqft"),
        )
        .orderBy(
            F.when(F.col("age_group") == "0-5 years", 1)
            .when(F.col("age_group") == "6-10 years", 2)
            .when(F.col("age_group") == "11-20 years", 3)
            .when(F.col("age_group") == "21-30 years", 4)
            .otherwise(5)
        )
    )
    age_analysis_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # F. Overall dataset statistics
    # [Distributed Whole-Dataset Aggregation & Percentile Approximation]
    # --------------------------------------------------------------------------
    print("\n[F] Overall System Statistics:")
    overall_stats_df = enriched_df.agg(
        F.count("*").alias("total_properties"),
        F.round(F.avg("rent"), 2).alias("avg_rent"),
        F.round(F.min("rent"), 2).alias("min_rent"),
        F.round(F.max("rent"), 2).alias("max_rent"),
        F.round(F.expr("percentile_approx(rent, 0.5)"), 2).alias("median_rent"),
        F.round(F.avg("area_sqft"), 1).alias("avg_area_sqft"),
        F.round(F.avg("rent_per_sqft"), 2).alias("avg_rent_per_sqft"),
    )
    overall_stats_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # G. Data quality & validation checks
    # [Distributed Filtering & Anomaly Counts]
    # --------------------------------------------------------------------------
    print("\n[G] Distributed Data Quality Checks:")
    quality_df = enriched_df.agg(
        # Null checks
        F.sum(F.when(F.col("zone").isNull(), 1).otherwise(0)).alias("null_zone_count"),
        F.sum(F.when(F.col("area_sqft").isNull(), 1).otherwise(0)).alias("null_area_count"),
        F.sum(F.when(F.col("bedrooms").isNull(), 1).otherwise(0)).alias("null_bedrooms_count"),
        F.sum(F.when(F.col("bathrooms").isNull(), 1).otherwise(0)).alias("null_bathrooms_count"),
        F.sum(F.when(F.col("furnished").isNull(), 1).otherwise(0)).alias("null_furnished_count"),
        F.sum(F.when(F.col("age_years").isNull(), 1).otherwise(0)).alias("null_age_count"),
        F.sum(F.when(F.col("rent").isNull(), 1).otherwise(0)).alias("null_rent_count"),
        # Domain validation checks
        F.sum(F.when(F.col("area_sqft") <= 0, 1).otherwise(0)).alias("invalid_area_count"),
        F.sum(F.when(F.col("bedrooms") <= 0, 1).otherwise(0)).alias("invalid_bedrooms_count"),
        F.sum(F.when(F.col("bathrooms") <= 0, 1).otherwise(0)).alias("invalid_bathrooms_count"),
        F.sum(F.when(F.col("rent") <= 0, 1).otherwise(0)).alias("invalid_rent_count"),
        F.sum(F.when(~F.col("furnished").isin([0, 1]), 1).otherwise(0)).alias("invalid_furnished_count"),
        F.sum(F.when(F.col("age_years") < 0, 1).otherwise(0)).alias("invalid_negative_age_count"),
    )
    quality_df.show(truncate=False)

    # --------------------------------------------------------------------------
    # 5. Save all analytics outputs back to HDFS in Parquet format
    # [Distributed Write to HDFS DataNodes with Coalesce for Clean Dimension Tables]
    # --------------------------------------------------------------------------
    print("\n[5/8] Saving Analytics Results to HDFS...")
    destinations = {
        "zone_analysis": (zone_analysis_df, f"{analytics_base_hdfs}/zone_analysis"),
        "rent_per_sqft": (rent_psf_df, f"{analytics_base_hdfs}/rent_per_sqft"),
        "bhk_analysis": (bhk_analysis_df, f"{analytics_base_hdfs}/bhk_analysis"),
        "furnished_analysis": (furnished_analysis_df, f"{analytics_base_hdfs}/furnished_analysis"),
        "age_analysis": (age_analysis_df, f"{analytics_base_hdfs}/age_analysis"),
        "overall_statistics": (overall_stats_df, f"{analytics_base_hdfs}/overall_statistics"),
        "data_quality": (quality_df, f"{analytics_base_hdfs}/data_quality"),
    }

    for name, (table_df, hdfs_out) in destinations.items():
        t_w = time.time()
        # Coalesce to 1 partition for compact dimension summary storage in HDFS
        table_df.coalesce(1).write.mode("overwrite").parquet(hdfs_out)
        print(f"  - {name:20s} -> {hdfs_out} [SAVED in {time.time() - t_w:.2f}s]")

    elapsed = time.time() - start_time
    print(f"\n[8/8] Distributed Rental Analytics Complete in {elapsed:.2f} seconds!")
    print("=" * 80 + "\n")

    spark.stop()
    return {
        "total_records": total_records,
        "elapsed_seconds": elapsed,
        "destinations": list(destinations.keys()),
    }


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "hdfs://namenode:9000/rental_data/raw/mumbai_rentals_large"
    out = sys.argv[2] if len(sys.argv) > 2 else "hdfs://namenode:9000/rental_data/analytics"
    run_rental_analytics(inp, out)
