"""
HDFS Client Module for Rental Price Analyzer.
Interacts with Hadoop HDFS via the WebHDFS REST API (NameNode port 9870).
Includes automatic local simulation fallback for seamless local development without breaking Flask.
"""

import os
import io
import re
import shutil
import logging
from typing import List, Dict, Any, Optional
import requests
import pandas as pd

from .config import get_config, BigDataConfig

logger = logging.getLogger("HDFSClient")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class HDFSClient:
    """
    Client for managing files and directories on Hadoop HDFS.
    Uses WebHDFS HTTP protocol for 100% native OS portability (no winutils.exe required on Windows).
    Falls back gracefully to a local directory structure if HDFS is unavailable.
    """

    def __init__(self, config: Optional[BigDataConfig] = None):
        self.config = config or get_config()
        self.webhdfs_url = self.config.WEBHDFS_URL
        self.user = self.config.HDFS_USER
        self.timeout = self.config.HDFS_TIMEOUT_SECONDS
        self._connected = False
        self._check_connection()

    def _check_connection(self) -> bool:
        """Probe the NameNode WebHDFS status endpoint."""
        if self.config.STORAGE_MODE == "LOCAL":
            logger.info("Storage mode set to LOCAL. Using local simulated HDFS directory.")
            self._connected = False
            return False

        try:
            url = f"{self.webhdfs_url}/?op=GETFILESTATUS&user.name={self.user}"
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                self._connected = True
                logger.info(f"Successfully connected to Hadoop HDFS at {self.config.HDFS_HOST}:{self.config.HDFS_WEBHDFS_PORT}")
                return True
            else:
                logger.warning(f"HDFS responded with status {response.status_code}. Falling back to local mode.")
                self._connected = False
                return False
        except Exception as e:
            logger.warning(f"Could not connect to Hadoop HDFS ({e}). Local fallback mode active.")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _get_local_sim_path(self, hdfs_path: str) -> str:
        """Convert an HDFS virtual path into a local simulation filesystem path."""
        clean_path = hdfs_path.lstrip("/\\")
        full_path = os.path.join(self.config.LOCAL_SIMULATION_DIR, clean_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return full_path

    def _fix_datanode_url(self, location: str) -> str:
        """Map internal Docker container hostnames in WebHDFS redirects to host accessible URL."""
        if not location:
            return location
        pattern = rf"http://[^:/]+:{self.config.HDFS_DATANODE_PORT}"
        replacement = f"http://{self.config.HDFS_HOST}:{self.config.HDFS_DATANODE_PORT}"
        return re.sub(pattern, replacement, location)

    # --------------------------------------------------------------------------
    # Directory Management
    # --------------------------------------------------------------------------
    def mkdir(self, hdfs_path: str) -> bool:
        """Create a directory on HDFS or local fallback."""
        clean_path = "/" + hdfs_path.strip("/\\")
        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=MKDIRS&permission=777&user.name={self.user}"
            try:
                res = requests.put(url, timeout=self.timeout)
                return res.status_code == 200 and res.json().get("boolean", False)
            except Exception as e:
                logger.error(f"Failed to create HDFS dir {clean_path}: {e}")
                return False
        else:
            local_dir = self._get_local_sim_path(clean_path)
            os.makedirs(local_dir, exist_ok=True)
            return True

    def init_directories(self) -> Dict[str, bool]:
        """
        Initialize standard project directories on HDFS:
          - /rental_data/raw
          - /rental_data/analytics
          - /rental_data/predictions
          - /rental_data/raw/mumbai_rentals_large
        """
        dirs = [
            self.config.HDFS_BASE_DIR,
            self.config.HDFS_RAW_DIR,
            self.config.HDFS_ANALYTICS_DIR,
            self.config.HDFS_PREDICTIONS_DIR,
            self.config.HDFS_LARGE_RAW_DIR,
        ]
        results = {}
        for d in dirs:
            success = self.mkdir(d)
            results[d] = success
            status = "CREATED/EXISTS" if success else "FAILED"
            mode = "HDFS" if self._connected else "LOCAL SIMULATION"
            logger.info(f"[{mode}] Directory {d} -> {status}")
        return results

    # --------------------------------------------------------------------------
    # File Operations
    # --------------------------------------------------------------------------
    def exists(self, hdfs_path: str) -> bool:
        """Check if an HDFS file or directory exists."""
        clean_path = "/" + hdfs_path.strip("/\\")
        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=GETFILESTATUS&user.name={self.user}"
            try:
                res = requests.get(url, timeout=self.timeout)
                return res.status_code == 200
            except Exception:
                return False
        else:
            return os.path.exists(self._get_local_sim_path(clean_path))

    def upload_file(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> bool:
        """
        Upload a single file from local filesystem to HDFS.
        Implements WebHDFS two-step upload protocol with Docker network host normalization.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        clean_path = "/" + hdfs_path.strip("/\\")

        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=CREATE&overwrite={str(overwrite).lower()}&user.name={self.user}"
            try:
                init_res = requests.put(url, allow_redirects=False, timeout=self.timeout)
                if init_res.status_code == 307:
                    datanode_location = self._fix_datanode_url(init_res.headers.get("Location", ""))
                    if not datanode_location:
                        logger.error("No DataNode location returned in WebHDFS 307 redirect.")
                        return False

                    with open(local_path, "rb") as f:
                        put_res = requests.put(
                            datanode_location,
                            data=f,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=180,
                        )
                    return put_res.status_code == 201
                elif init_res.status_code == 201:
                    return True
                else:
                    logger.error(f"HDFS create request failed: {init_res.status_code} {init_res.text}")
                    return False
            except Exception as e:
                logger.error(f"Upload to HDFS failed: {e}. Attempting local simulation.")
                target = self._get_local_sim_path(clean_path)
                shutil.copy2(local_path, target)
                return True
        else:
            target = self._get_local_sim_path(clean_path)
            shutil.copy2(local_path, target)
            logger.info(f"[LOCAL] Stored {local_path} at simulated path: {target}")
            return True

    def upload_dir(self, local_dir: str, hdfs_dir: str, overwrite: bool = True) -> Dict[str, bool]:
        """
        Upload an entire local directory (all files) to a target HDFS directory.
        Returns dictionary of {filename: success_status}.
        """
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(f"Local directory not found: {local_dir}")

        clean_hdfs_dir = "/" + hdfs_dir.strip("/\\")
        self.mkdir(clean_hdfs_dir)

        results = {}
        files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
        for fname in sorted(files):
            loc_file = os.path.join(local_dir, fname)
            hdfs_file = f"{clean_hdfs_dir}/{fname}"
            ok = self.upload_file(loc_file, hdfs_file, overwrite=overwrite)
            results[fname] = ok
            logger.info(f"Uploaded {fname} -> {hdfs_file} [{'OK' if ok else 'FAIL'}]")
        return results

    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """Download a file from HDFS to local filesystem."""
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        clean_path = "/" + hdfs_path.strip("/\\")
        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=OPEN&user.name={self.user}"
            try:
                res = requests.get(url, stream=True, allow_redirects=False, timeout=60)
                if res.status_code == 307:
                    dn_url = self._fix_datanode_url(res.headers.get("Location", ""))
                    stream_res = requests.get(dn_url, stream=True, timeout=180)
                    if stream_res.status_code == 200:
                        with open(local_path, "wb") as f:
                            for chunk in stream_res.iter_content(chunk_size=65536):
                                f.write(chunk)
                        return True
                elif res.status_code == 200:
                    with open(local_path, "wb") as f:
                        for chunk in res.iter_content(chunk_size=65536):
                            f.write(chunk)
                    return True
                return False
            except Exception as e:
                logger.error(f"Download from HDFS failed: {e}")
                return False
        else:
            sim_path = self._get_local_sim_path(clean_path)
            if not os.path.exists(sim_path):
                return False
            shutil.copy2(sim_path, local_path)
            return True

    def list_dir(self, hdfs_path: str) -> List[Dict[str, Any]]:
        """List files and subdirectories within an HDFS path."""
        clean_path = "/" + hdfs_path.strip("/\\")
        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=LISTSTATUS&user.name={self.user}"
            try:
                res = requests.get(url, timeout=self.timeout)
                if res.status_code == 200:
                    statuses = res.json().get("FileStatuses", {}).get("FileStatus", [])
                    return [
                        {
                            "name": item["pathSuffix"],
                            "type": item["type"],
                            "length": item["length"],
                            "modificationTime": item["modificationTime"],
                        }
                        for item in statuses
                    ]
                return []
            except Exception as e:
                logger.error(f"Failed to list HDFS dir {clean_path}: {e}")
                return []
        else:
            sim_dir = self._get_local_sim_path(clean_path)
            if not os.path.exists(sim_dir):
                return []
            results = []
            for entry in os.scandir(sim_dir):
                results.append({
                    "name": entry.name,
                    "type": "DIRECTORY" if entry.is_dir() else "FILE",
                    "length": entry.stat().st_size if entry.is_file() else 0,
                    "modificationTime": int(entry.stat().st_mtime * 1000),
                })
            return results

    # --------------------------------------------------------------------------
    # Pandas DataFrame Helpers
    # --------------------------------------------------------------------------
    def read_csv(self, hdfs_path: str) -> pd.DataFrame:
        """Read a CSV file from HDFS into a Pandas DataFrame."""
        clean_path = "/" + hdfs_path.strip("/\\")
        if self._connected:
            url = f"{self.webhdfs_url}{clean_path}?op=OPEN&user.name={self.user}"
            res = requests.get(url, allow_redirects=False, timeout=60)
            if res.status_code == 307:
                dn_url = self._fix_datanode_url(res.headers.get("Location", ""))
                res = requests.get(dn_url, timeout=120)
            if res.status_code == 200:
                return pd.read_csv(io.StringIO(res.text))
            raise IOError(f"Failed to read HDFS CSV at {clean_path}: HTTP {res.status_code}")
        else:
            sim_path = self._get_local_sim_path(clean_path)
            if not os.path.exists(sim_path):
                raise FileNotFoundError(f"Simulated HDFS file not found: {sim_path}")
            return pd.read_csv(sim_path)


# Global client instance
_client_instance = None


def get_hdfs_client() -> HDFSClient:
    """Return the global HDFSClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = HDFSClient()
    return _client_instance


if __name__ == "__main__":
    client = get_hdfs_client()
    print("\n--- HDFS Status Diagnostic ---")
    print(f"Connection status: {'CONNECTED (Cluster)' if client.is_connected else 'LOCAL SIMULATION (Fallback)'}")
    print(f"HDFS WebHDFS URL:  {client.webhdfs_url}")
    print("\nInitializing project directories...")
    dirs = client.init_directories()
    for d, status in dirs.items():
        print(f"  {d}: {'OK' if status else 'FAILED'}")
    print("\nListing root directories:")
    for item in client.list_dir("/rental_data"):
        print(f"  [{item['type']}] {item['name']} ({item['length']} bytes)")
