#!/usr/bin/env bash
# Helper script to execute Distributed Analytics inside Spark container
set -e

echo "=== Running Distributed Rental Analytics Job on Spark Cluster ==="
/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  /opt/spark/work-dir/rental_analytics.py \
  hdfs://namenode:9000/rental_data/raw/mumbai_rentals_large \
  hdfs://namenode:9000/rental_data/analytics

echo ""
echo "=== Running Verification of Analytics Outputs in HDFS ==="
/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  /opt/spark/work-dir/verify_analytics.py \
  hdfs://namenode:9000/rental_data/analytics
