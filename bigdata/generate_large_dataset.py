"""
Scalable Large Dataset Generator for Rental Price Analyzer.
Generates ~1,000,000 realistic rental property records based on data/mumbai_rentals.csv
and outputs partitioned Apache Parquet files without high memory consumption.
"""

import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import get_config


def generate_large_dataset(
    source_csv: str = "data/mumbai_rentals.csv",
    output_dir: str = "data/bigdata/mumbai_rentals_large",
    total_records: int = 1_000_000,
    num_partitions: int = 10,
    random_seed: int = 42,
) -> dict:
    """
    Generate ~1,000,000 realistic Mumbai rental records partitioned across Parquet files.
    Learns statistical parameters from source_csv to preserve realistic distributions.
    """
    start_time = time.time()
    np.random.seed(random_seed)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source dataset not found: {source_csv}")

    print(f"\n[1/4] Reading baseline statistical parameters from {source_csv}...")
    base_df = pd.read_csv(source_csv)

    # Validate required columns
    required_cols = ["zone", "area_sqft", "bedrooms", "bathrooms", "furnished", "age_years", "rent"]
    for col in required_cols:
        if col not in base_df.columns:
            raise ValueError(f"Required column '{col}' missing from source dataset.")

    # Calculate zone distribution probabilities from baseline data
    zone_counts = base_df["zone"].value_counts(normalize=True)
    zones = list(zone_counts.index)
    zone_probs = list(zone_counts.values)

    # Calculate empirical price-per-sqft (PSF) mean and std for each zone
    base_df["psf"] = base_df["rent"] / (base_df["area_sqft"] * (1 + 0.10 * base_df["furnished"]))
    zone_psf_stats = {}
    for z in zones:
        z_slice = base_df[base_df["zone"] == z]["psf"]
        zone_psf_stats[z] = {
            "mean": float(z_slice.mean()),
            "std": float(z_slice.std()),
            "min_psf": float(z_slice.quantile(0.01)),
            "max_psf": float(z_slice.quantile(0.99)),
        }

    print(f"  Zones identified: {zones}")
    for z, st in zone_psf_stats.items():
        print(f"    - {z}: Mean PSF = Rs.{st['mean']:.2f}, Std = {st['std']:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    records_per_partition = total_records // num_partitions
    remainder = total_records % num_partitions

    print(f"\n[2/4] Generating {total_records:,} records into {num_partitions} Parquet partitions...")
    print(f"  Target directory: {output_dir}")

    parquet_schema = pa.schema([
        ("zone", pa.string()),
        ("area_sqft", pa.int64()),
        ("bedrooms", pa.int64()),
        ("bathrooms", pa.int64()),
        ("furnished", pa.int64()),
        ("age_years", pa.int64()),
        ("rent", pa.float64()),
    ])

    total_generated = 0
    partition_files = []

    for p in range(num_partitions):
        chunk_n = records_per_partition + (1 if p < remainder else 0)
        p_start = time.time()

        # 1. Zone assignment according to empirical distribution
        p_zones = np.random.choice(zones, size=chunk_n, p=zone_probs)

        # 2. Area sqft: Log-normal / Uniform mixture matching Mumbai distribution (300 to 2000 sqft)
        p_area = np.random.randint(300, 2000, size=chunk_n)

        # 3. Bedrooms: Strongly correlated with area
        # 300-600 sqft -> 1 BHK, 600-1000 -> 2 BHK, 1000-1500 -> 3 BHK, 1500+ -> 4 or 5 BHK
        raw_beds = np.round(p_area / 400.0).astype(int)
        p_bedrooms = np.clip(raw_beds, 1, 5)

        # 4. Bathrooms: Correlated with bedrooms (equal or 1 less, minimum 1)
        bath_offset = np.random.choice([0, 1], size=chunk_n, p=[0.65, 0.35])
        p_bathrooms = np.clip(p_bedrooms - bath_offset, 1, 5)

        # 5. Furnished: 0 (70%) or 1 (30%)
        p_furnished = np.random.choice([0, 1], size=chunk_n, p=[0.70, 0.30])

        # 6. Age: 0 to 40 years
        p_age = np.random.randint(0, 40, size=chunk_n)

        # 7. Rent calculation with realistic compounding factors:
        # base_psf per zone + furnishing premium (10%) + age depreciation (-0.2%/yr) + market noise (±15%)
        p_psf = np.array([zone_psf_stats[z]["mean"] for z in p_zones])
        # Add realistic variability around zone mean
        psf_noise = np.random.normal(1.0, 0.08, size=chunk_n)
        effective_psf = np.clip(p_psf * psf_noise, 30.0, 350.0)

        p_rent = p_area * effective_psf
        # Furnishing status adds 10% premium
        p_rent = p_rent * (1.0 + p_furnished * 0.10)
        # Newer buildings command modest premium over older ones (max ~8% difference across 40 yrs)
        age_factor = 1.0 - (p_age * 0.002)
        p_rent = p_rent * np.clip(age_factor, 0.90, 1.0)
        # General market randomness
        market_fluctuation = np.random.uniform(0.92, 1.08, size=chunk_n)
        p_rent = np.round(p_rent * market_fluctuation, 2)

        # Guard against any negative or unrealistic values
        p_rent = np.maximum(p_rent, 10000.0)

        # Build PyArrow Table directly (fast, zero-copy, minimal memory footprint)
        table = pa.Table.from_arrays(
            [
                pa.array(p_zones, type=pa.string()),
                pa.array(p_area, type=pa.int64()),
                pa.array(p_bedrooms, type=pa.int64()),
                pa.array(p_bathrooms, type=pa.int64()),
                pa.array(p_furnished, type=pa.int64()),
                pa.array(p_age, type=pa.int64()),
                pa.array(p_rent, type=pa.float64()),
            ],
            schema=parquet_schema,
        )

        part_filename = f"part-{p:05d}.parquet"
        part_path = os.path.join(output_dir, part_filename)

        # Write Parquet with Snappy compression
        pq.write_table(table, part_path, compression="snappy")

        part_size_mb = os.path.getsize(part_path) / (1024 * 1024)
        total_generated += chunk_n
        partition_files.append(part_path)

        p_duration = time.time() - p_start
        print(f"  [Partition {p+1:02d}/{num_partitions:02d}] {part_filename}: {chunk_n:,} records ({part_size_mb:.2f} MB) in {p_duration:.2f}s")

    total_size_bytes = sum(os.path.getsize(f) for f in partition_files)
    total_size_mb = total_size_bytes / (1024 * 1024)
    elapsed = time.time() - start_time

    print(f"\n[3/4] Generation Complete:")
    print(f"  - Total records:    {total_generated:,}")
    print(f"  - Parquet files:    {len(partition_files)}")
    print(f"  - Total disk size:  {total_size_mb:.2f} MB ({total_size_bytes:,} bytes)")
    print(f"  - Execution time:   {elapsed:.2f} seconds")

    return {
        "total_records": total_generated,
        "partitions": len(partition_files),
        "total_size_bytes": total_size_bytes,
        "total_size_mb": total_size_mb,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    cfg = get_config()
    res = generate_large_dataset(
        source_csv="data/mumbai_rentals.csv",
        output_dir=cfg.LOCAL_LARGE_DATASET_DIR,
        total_records=1_000_000,
        num_partitions=10,
        random_seed=42,
    )
