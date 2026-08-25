"""
Simple validation script for file_downloader.DownloadManager using LocalProvider.
Creates a temporary source folder with dummy .pth files, runs DownloadManager, and verifies files copied correctly.
Run with: python -m tests.validate_file_downloader or python tests/validate_file_downloader.py
"""
import os
import tempfile
import shutil
import sys

# Ensure repo root is on sys.path so imports work when running script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vispr.tools.common.file_downloader import DownloadManager


def create_dummy_pth(dirpath, name, content=b'dummy'):
    path = os.path.join(dirpath, name)
    with open(path, 'wb') as f:
        f.write(content)
    return path


def main():
    src_dir = tempfile.mkdtemp(prefix='fd_src_')
    dst_dir = tempfile.mkdtemp(prefix='fd_dst_')
    try:
        # create dummy files
        files = ['model_a.pth', 'model_b.pth', 'readme.txt']
        contents = {}
        for f in files:
            c = (f + '_content').encode('utf-8')
            p = create_dummy_pth(src_dir, f, content=c)
            contents[os.path.basename(p)] = c

        # Run downloader to fetch only '*.pth'
        dm = DownloadManager(provider='local')
        downloaded = dm.download_checkpoints(folder=src_dir, dest_dir=dst_dir, pattern='*.pth')
        print('Downloaded paths:', downloaded)

        # Expect two files
        expected = {'model_a.pth', 'model_b.pth'}
        got = {os.path.basename(p) for p in downloaded}
        if expected != got:
            print('TEST FAILED: Expected files', expected, 'but got', got)
            sys.exit(2)

        # Verify content
        for p in downloaded:
            bn = os.path.basename(p)
            with open(p, 'rb') as f:
                data = f.read()
            if data != contents[bn]:
                print('TEST FAILED: content mismatch for', bn)
                sys.exit(3)

        print('TEST PASSED: all files downloaded and verified')
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(dst_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
