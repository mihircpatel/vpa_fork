"""
file_downloader.py

Reusable utility to download checkpoint files (*.pth) from different storage providers.
- Implements a StorageProvider interface and several providers: Local, HTTP, GoogleColab (gdown), S3, Azure.
- Exposes DownloadManager for simple integration.
- Provides meaningful errors and logging. Atomic file writes are used to avoid partial files.

Usage (as module):
from file_downloader import DownloadManager
mgr = DownloadManager(provider='local', provider_config={'folder': './checkpoints'})
mgr.download_checkpoints(dest_dir='./checkpoints_local')

Usage (CLI):
python file_downloader.py --provider local --folder ./checkpoints --dest ./out

"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional
from vispr.tools.common.logger import get_logger

try:
    import requests
except Exception:
    requests = None  # type: ignore

# Optional providers
try:
    import gdown
except Exception:
    gdown = None  # type: ignore

try:
    import boto3
except Exception:
    boto3 = None  # type: ignore

try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None  # type: ignore


# Default module logger (will be overridden by DownloadManager if requested)
logger = get_logger('file_downloader')


class StorageProviderError(Exception):
    pass


class StorageProvider(ABC):
    """Abstract interface for storage providers."""

    @abstractmethod
    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        """Return a list of file identifiers (paths/URLs) matching pattern under folder."""

    @abstractmethod
    def download(self, file_id: str, dest_path: str) -> None:
        """Download `file_id` to local path `dest_path`.
        Implementations must write atomically into dest_path (write to a temp and os.replace).
        """


class LocalProvider(StorageProvider):
    """Provider for files available on the local filesystem (e.g., Google Colab mounted drive).

    Config:
      - None required. Provide `folder` to DownloadManager.
    """

    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        if not os.path.isdir(folder):
            raise StorageProviderError(f"Local folder does not exist: {folder}")
        files = [os.path.join(folder, f) for f in os.listdir(folder) if fnmatch.fnmatch(f, pattern)]
        logger.debug("LocalProvider.list_files -> %s files", len(files))
        return files

    def download(self, file_id: str, dest_path: str) -> None:
        # file_id is a local path
        if not os.path.exists(file_id):
            raise StorageProviderError(f"Source file not found: {file_id}")
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        # Copy using a temp file then replace for atomicity
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest_path) or '.', prefix='.tmp_dl_', suffix='.pth') as tmpf:
            tmpname = tmpf.name
        try:
            shutil.copy2(file_id, tmpname)
            os.replace(tmpname, dest_path)
            logger.info("Copied %s -> %s", file_id, dest_path)
        except Exception as e:
            try:
                os.remove(tmpname)
            except Exception:
                pass
            raise StorageProviderError(f"Failed to copy {file_id} to {dest_path}: {e}")


class HTTPProvider(StorageProvider):
    """Provider for direct HTTP/HTTPS URLs. The folder argument is ignored; list_files expects a list of URLs in provider config."""

    def __init__(self, config: Optional[Dict] = None):
        if requests is None:
            raise StorageProviderError("requests library is required for HTTPProvider. Install via `pip install requests`.")
        self.config = config or {}

    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        # Expect config to contain 'urls' list OR folder to be a newline-separated URL file
        urls = self.config.get('urls')
        if urls:
            return [u for u in urls if fnmatch.fnmatch(os.path.basename(u), pattern)]
        # Try reading folder as file containing URLs
        if os.path.isfile(folder):
            with open(folder, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            return [u for u in lines if fnmatch.fnmatch(os.path.basename(u), pattern)]
        raise StorageProviderError("HTTPProvider needs either 'urls' in config or a file path containing URLs")

    def download(self, file_id: str, dest_path: str) -> None:
        logger.debug("HTTPProvider.download %s -> %s", file_id, dest_path)
        resp = requests.get(file_id, stream=True, timeout=30)
        if resp.status_code != 200:
            raise StorageProviderError(f"Failed to download {file_id}: HTTP {resp.status_code}")
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest_path) or '.', prefix='.tmp_dl_', suffix='.pth') as tmpf:
            tmpname = tmpf.name
            try:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        tmpf.write(chunk)
                tmpf.flush()
                os.fsync(tmpf.fileno())
                os.replace(tmpname, dest_path)
                logger.info("Downloaded %s -> %s", file_id, dest_path)
            except Exception as e:
                try:
                    os.remove(tmpname)
                except Exception:
                    pass
                raise StorageProviderError(f"Failed to save {file_id} to {dest_path}: {e}")


class GoogleColabProvider(StorageProvider):
    """Provider for Google Colab / Google Drive.

    This implementation supports two modes:
      - local: treat `folder` as a local path (e.g., /content/drive/MyDrive/checkpoints)
      - drive_ids: provider_config must include {'drive_ids': {basename: file_id}} and will use gdown to download by id

    gdown is optional; if not installed and only local mode is required, it will still work.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        # If drive_ids provided, return mapped URLs (gdown accepts id parameter)
        drive_ids = self.config.get('drive_ids')
        if drive_ids:
            matches = [v for k, v in drive_ids.items() if fnmatch.fnmatch(k, pattern)]
            return matches
        # Otherwise, assume local folder path (mounted drive)
        lp = folder
        if os.path.isdir(lp):
            return [os.path.join(lp, f) for f in os.listdir(lp) if fnmatch.fnmatch(f, pattern)]
        raise StorageProviderError(f"GoogleColabProvider: no drive_ids in config and folder not found: {folder}")

    def download(self, file_id: str, dest_path: str) -> None:
        # file_id may be a local path or a Google Drive file id
        if os.path.exists(file_id):
            # local path
            LocalProvider().download(file_id, dest_path)
            return
        if gdown is None:
            raise StorageProviderError("gdown is required to download from Google Drive ids. Install via `pip install gdown`.")
        # gdown accepts either id or full url; pass file_id directly
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest_path) or '.', prefix='.tmp_dl_', suffix='.pth') as tmpf:
            tmpname = tmpf.name
        try:
            # gdown.download(url, output) returns output path
            # The quiet flag suppressed; let gdown log as needed
            gdown.download(file_id, tmpname, quiet=True)
            os.replace(tmpname, dest_path)
            logger.info("gdown: %s -> %s", file_id, dest_path)
        except Exception as e:
            try:
                os.remove(tmpname)
            except Exception:
                pass
            raise StorageProviderError(f"gdown failed to download {file_id}: {e}")


class S3Provider(StorageProvider):
    """Provider for AWS S3. Requires boto3 and appropriate AWS credentials to be configured.

    Config:
      - bucket: required
      - prefix: optional
    """

    def __init__(self, config: Optional[Dict] = None):
        if boto3 is None:
            raise StorageProviderError("boto3 is required for S3Provider. Install via `pip install boto3`.")
        self.config = config or {}
        self.s3 = boto3.client('s3')

    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        bucket = self.config.get('bucket')
        prefix = self.config.get('prefix', folder or '')
        if not bucket:
            raise StorageProviderError('S3Provider requires `bucket` in config')
        paginator = self.s3.get_paginator('list_objects_v2')
        keys: List[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if fnmatch.fnmatch(os.path.basename(key), pattern):
                    keys.append(key)
        return [f's3://{bucket}/{k}' for k in keys]

    def download(self, file_id: str, dest_path: str) -> None:
        # file_id like s3://bucket/key
        if not file_id.startswith('s3://'):
            raise StorageProviderError('S3Provider.download expects s3://bucket/key')
        _, _, rest = file_id.partition('s3://')
        bucket, _, key = rest.partition('/')
        if not bucket or not key:
            raise StorageProviderError(f'Invalid S3 path: {file_id}')
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        tmpf = tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest_path) or '.', prefix='.tmp_dl_', suffix='.pth')
        tmpname = tmpf.name
        tmpf.close()
        try:
            self.s3.download_file(bucket, key, tmpname)
            os.replace(tmpname, dest_path)
            logger.info('S3 downloaded %s -> %s', file_id, dest_path)
        except Exception as e:
            try:
                os.remove(tmpname)
            except Exception:
                pass
            raise StorageProviderError(f'S3 download failed for {file_id}: {e}')


class AzureProvider(StorageProvider):
    """Provider for Azure Blob Storage. Requires azure-storage-blob package and connection string/credentials in config."""

    def __init__(self, config: Optional[Dict] = None):
        if BlobServiceClient is None:
            raise StorageProviderError("azure-storage-blob is required for AzureProvider. Install via `pip install azure-storage-blob`.")
        self.config = config or {}
        conn_str = self.config.get('connection_string')
        if not conn_str:
            raise StorageProviderError('AzureProvider requires `connection_string` in config')
        self.client = BlobServiceClient.from_connection_string(conn_str)

    def list_files(self, folder: str, pattern: str = "*.pth") -> List[str]:
        container = self.config.get('container')
        prefix = self.config.get('prefix', folder or '')
        if not container:
            raise StorageProviderError('AzureProvider requires `container` in config')
        container_client = self.client.get_container_client(container)
        matches: List[str] = []
        for blob in container_client.list_blobs(name_starts_with=prefix):
            if fnmatch.fnmatch(os.path.basename(blob.name), pattern):
                matches.append(f"azure://{container}/{blob.name}")
        return matches

    def download(self, file_id: str, dest_path: str) -> None:
        # file_id like azure://container/path
        if not file_id.startswith('azure://'):
            raise StorageProviderError('AzureProvider.download expects azure://container/blobpath')
        _, _, rest = file_id.partition('azure://')
        container, _, blobpath = rest.partition('/')
        if not container or not blobpath:
            raise StorageProviderError(f'Invalid Azure path: {file_id}')
        blob_client = self.client.get_blob_client(container=container, blob=blobpath)
        os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest_path) or '.', prefix='.tmp_dl_', suffix='.pth') as tmpf:
            tmpname = tmpf.name
        try:
            with open(tmpname, 'wb') as f:
                stream = blob_client.download_blob()
                stream.readinto(f)
            os.replace(tmpname, dest_path)
            logger.info('Azure downloaded %s -> %s', file_id, dest_path)
        except Exception as e:
            try:
                os.remove(tmpname)
            except Exception:
                pass
            raise StorageProviderError(f'Azure download failed for {file_id}: {e}')


_PROVIDER_MAP = {
    'local': LocalProvider,
    'http': HTTPProvider,
    'https': HTTPProvider,
    'gc': GoogleColabProvider,
    'google_colab': GoogleColabProvider,
    's3': S3Provider,
    'azure': AzureProvider,
}


class DownloadManager:
    """High-level manager to list and download checkpoint files.

    Example:
        mgr = DownloadManager(provider='local', provider_config={'folder': './checkpoints'})
        mgr.download_checkpoints(dest_dir='./out', pattern='*.pth')

    The DownloadManager accepts optional logging configuration to integrate with the project's
    get_logger helper (log_file, level, console, rotate).
    """

    def __init__(self, provider: str = 'local', provider_config: Optional[Dict] = None,
                 logger_obj: Optional[logging.Logger] = None, log_name: Optional[str] = 'file_downloader',
                 log_file: Optional[str] = None, log_level: Optional[str] = None, console: bool = True, rotate: bool = True):
        self.provider_name = provider
        self.provider_config = provider_config or {}
        # Prefer explicit logger_obj; otherwise configure via get_logger for consistent project logging
        if logger_obj is not None:
            self.logger = logger_obj
        else:
            # get_logger handles creating log dir, rotating handler and console handler
            self.logger = get_logger(log_name, log_file=log_file, level=log_level, console=console, rotate=rotate)

        provider_cls = _PROVIDER_MAP.get(provider)
        if provider_cls is None:
            raise StorageProviderError(f'Unknown provider: {provider}. Supported: {list(_PROVIDER_MAP.keys())}')
        # Some providers accept config in constructor
        try:
            self.provider: StorageProvider = provider_cls(self.provider_config) if self._accepts_config(provider_cls) else provider_cls()
        except TypeError:
            # fallback for provider constructors without config
            self.provider = provider_cls()

    def _accepts_config(self, cls) -> bool:
        # heuristic: check if __init__ accepts config param by name (simple and robust enough)
        try:
            varnames = cls.__init__.__code__.co_varnames
        except Exception:
            return False
        return 'config' in varnames

    def download_checkpoints(self, folder: str = '', dest_dir: str = '.', pattern: str = '*.pth', dry_run: bool = False) -> List[str]:
        """List and download checkpoints.

        Args:
            folder: provider-specific folder/path. For local provider this is the folder path.
            dest_dir: local directory where files will be downloaded.
            pattern: glob pattern to filter checkpoint filenames.
            dry_run: if True, only list matched files without downloading.

        Returns:
            List of local paths to downloaded files.
        """
        self.logger.info('DownloadManager: provider=%s folder=%s pattern=%s dest=%s', self.provider_name, folder, pattern, dest_dir)
        files = self.provider.list_files(folder, pattern)
        if not files:
            self.logger.warning('No files matched pattern %s in %s using provider %s', pattern, folder, self.provider_name)
            return []
        os.makedirs(dest_dir, exist_ok=True)
        downloaded: List[str] = []
        for fid in files:
            filename = os.path.basename(fid) if not fid.startswith(('s3://', 'azure://')) else os.path.basename(fid.split('/')[-1])
            dest_path = os.path.join(dest_dir, filename)
            if dry_run:
                self.logger.info('[dry-run] Would download %s -> %s', fid, dest_path)
                downloaded.append(dest_path)
                continue
            try:
                self.provider.download(fid, dest_path)
                downloaded.append(dest_path)
            except Exception as e:
                self.logger.error('Failed to download %s: %s', fid, e)
        return downloaded


# CLI integration
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download checkpoint files from various providers')
    parser.add_argument('--provider', default='local', help='Provider name: local, http, google_colab, s3, azure')
    parser.add_argument('--folder', default='', help='Folder or provider-specific path (for local this is local folder)')
    parser.add_argument('--dest', default='./downloaded_checkpoints', help='Local destination directory')
    parser.add_argument('--pattern', default='*.pth', help='Filename glob pattern to match')
    parser.add_argument('--provider-config', default=None, help='JSON string or path to JSON file with provider-specific configuration')
    parser.add_argument('--dry-run', action='store_true', help='Only list matched files without downloading')

    # Logging options to align with project's logger utility
    parser.add_argument('--log-name', default='file_downloader', help='Logger name to use')
    parser.add_argument('--log-file', default=None, help='Optional log file path (overrides default logs/<name>.log)')
    parser.add_argument('--log-level', default=None, help='Log level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--no-console', action='store_true', help='Disable console logging')
    parser.add_argument('--no-rotate', action='store_true', help='Disable rotating file handler')

    args = parser.parse_args()

    pconfig: Dict = {}
    if args.provider_config:
        # Try to parse as JSON string or path
        if os.path.isfile(args.provider_config):
            with open(args.provider_config, 'r') as f:
                pconfig = json.load(f)
        else:
            try:
                pconfig = json.loads(args.provider_config)
            except Exception as e:
                raise SystemExit(f'Failed to parse provider-config: {e}')

    dm = DownloadManager(provider=args.provider, provider_config=pconfig,
                         log_name=args.log_name, log_file=args.log_file, log_level=args.log_level,
                         console=(not args.no_console), rotate=(not args.no_rotate))
    try:
        results = dm.download_checkpoints(folder=args.folder, dest_dir=args.dest, pattern=args.pattern, dry_run=args.dry_run)
        if args.dry_run:
            print('Matched files:')
            for r in results:
                print('  ', r)
        else:
            print(f'Downloaded {len(results)} files to {args.dest}')
    except Exception as e:
        raise SystemExit(f'Error: {e}')
