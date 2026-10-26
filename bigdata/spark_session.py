"""
SparkSession factory and manager for Rental Price Analyzer.
Configures PySpark with HDFS integration, driver/executor memory, and graceful fallbacks.
"""

import os
import logging
from typing import Optional

from .config import get_config, BigDataConfig

logger = logging.getLogger("SparkSessionManager")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_spark_instance = None


def is_spark_available() -> bool:
    """Check if PySpark is installed and importable."""
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        return False


def get_spark_session(config: Optional[BigDataConfig] = None):
    """
    Get or create an active SparkSession configured for the Rental Price Analyzer.
    Connects to Hadoop HDFS if available, or operates in local filesystem mode.
    Returns None with a descriptive warning if PySpark or Java is unavailable.
    """
    global _spark_instance
    if _spark_instance is not None:
        return _spark_instance

    cfg = config or get_config()

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.warning(
            "PySpark is not installed in the current Python environment. "
            "Install it via 'pip install -r requirements_bigdata.txt'. "
            "Returning None for SparkSession."
        )
        return None

    try:
        logger.info(f"Initializing SparkSession [App: {cfg.SPARK_APP_NAME}, Master: {cfg.SPARK_MASTER}]...")

        builder = (
            SparkSession.builder.appName(cfg.SPARK_APP_NAME)
            .master(cfg.SPARK_MASTER)
            .config("spark.driver.memory", cfg.SPARK_DRIVER_MEMORY)
            .config("spark.executor.memory", cfg.SPARK_EXECUTOR_MEMORY)
            .config("spark.sql.shuffle.partitions", cfg.SPARK_SQL_SHUFFLE_PARTITIONS)
            .config("spark.ui.showConsoleProgress", "false")
        )

        # Configure HDFS defaultFS if running in cluster mode
        if cfg.STORAGE_MODE == "HDFS":
            builder = (
                builder.config("spark.hadoop.fs.defaultFS", cfg.HDFS_DEFAULT_FS)
                .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
            )

        _spark_instance = builder.getOrCreate()
        # Set log level to WARN to suppress verbose Spark info logs
        _spark_instance.sparkContext.setLogLevel("WARN")

        logger.info(f"SparkSession created successfully! Spark Version: {_spark_instance.version}")
        return _spark_instance

    except Exception as e:
        logger.error(
            f"Failed to create SparkSession: {e}. "
            "Ensure Java (JDK 8/11/17) is installed and JAVA_HOME is configured. "
            "Returning None."
        )
        _spark_instance = None
        return None


def stop_spark_session():
    """Cleanly stop the active SparkSession."""
    global _spark_instance
    if _spark_instance is not None:
        try:
            logger.info("Stopping active SparkSession...")
            _spark_instance.stop()
        except Exception as e:
            logger.warning(f"Error while stopping SparkSession: {e}")
        finally:
            _spark_instance = None


if __name__ == "__main__":
    print("\n--- Spark Environment Diagnostic ---")
    available = is_spark_available()
    print(f"PySpark library available: {available}")
    if available:
        spark = get_spark_session()
        if spark:
            print(f"Spark session active: {spark.version}")
            print(f"Spark Master:         {spark.conf.get('spark.master')}")
            stop_spark_session()
        else:
            print("Spark session could not be created. Check Java installation.")
    else:
        print("Run 'pip install -r requirements_bigdata.txt' to install PySpark.")
