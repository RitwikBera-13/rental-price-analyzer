"""
Central configuration module for Big Data components (Hadoop HDFS & Apache Spark).
Supports environment variables with sensible defaults for local and containerized setups.
"""

import os
from dataclasses import dataclass, field


@dataclass
class BigDataConfig:
    # --------------------------------------------------------------------------
    # General / Runtime Mode
    # --------------------------------------------------------------------------
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "HDFS").upper()

    # --------------------------------------------------------------------------
    # Hadoop HDFS Configuration
    # --------------------------------------------------------------------------
    HDFS_HOST: str = os.getenv("HDFS_HOST", "localhost")
    HDFS_RPC_PORT: int = int(os.getenv("HDFS_RPC_PORT", "9000"))
    HDFS_WEBHDFS_PORT: int = int(os.getenv("HDFS_WEBHDFS_PORT", "9870"))
    HDFS_DATANODE_PORT: int = int(os.getenv("HDFS_DATANODE_PORT", "9864"))
    HDFS_USER: str = os.getenv("HDFS_USER", "root")

    # WebHDFS REST API Base URL
    @property
    def WEBHDFS_URL(self) -> str:
        return f"http://{self.HDFS_HOST}:{self.HDFS_WEBHDFS_PORT}/webhdfs/v1"

    # Hadoop Default FS URI (used by Spark & IPC)
    @property
    def HDFS_DEFAULT_FS(self) -> str:
        return f"hdfs://{self.HDFS_HOST}:{self.HDFS_RPC_PORT}"

    # --------------------------------------------------------------------------
    # HDFS Directory Hierarchy
    # --------------------------------------------------------------------------
    HDFS_BASE_DIR: str = "/rental_data"
    HDFS_RAW_DIR: str = "/rental_data/raw"
    HDFS_ANALYTICS_DIR: str = "/rental_data/analytics"
    HDFS_PREDICTIONS_DIR: str = "/rental_data/predictions"
    HDFS_LARGE_RAW_DIR: str = "/rental_data/raw/mumbai_rentals_large"

    # Local paths
    LOCAL_BIGDATA_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "bigdata",
    )
    LOCAL_LARGE_DATASET_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "bigdata",
        "mumbai_rentals_large",
    )
    LOCAL_SIMULATION_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "hdfs_sim",
    )

    # --------------------------------------------------------------------------
    # Apache Spark / PySpark Configuration
    # --------------------------------------------------------------------------
    SPARK_APP_NAME: str = os.getenv("SPARK_APP_NAME", "RentalPriceAnalyzer-BigData")
    SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
    SPARK_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "2g")
    SPARK_EXECUTOR_MEMORY: str = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
    SPARK_SQL_SHUFFLE_PARTITIONS: str = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "4")

    # Connection timeouts
    HDFS_TIMEOUT_SECONDS: int = int(os.getenv("HDFS_TIMEOUT_SECONDS", "10"))


# Global singleton instance
_config_instance = None


def get_config() -> BigDataConfig:
    """Return the global BigDataConfig singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = BigDataConfig()
    return _config_instance
