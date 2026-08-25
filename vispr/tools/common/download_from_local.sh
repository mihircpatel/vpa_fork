#!/usr/bin/env bash
# download_from_local.sh
# Convenience script to download .pth checkpoints from a local folder using file_downloader.py CLI.
# Place this file in vispr/tools/common and run from a shell.
#
# Usage:
#   ./download_from_local.sh "/absolute/path/to/checkpoints" "/absolute/path/to/dest" [--dry-run]
# Example:
#   ./download_from_local.sh "/home/user/checkpoints" "$HOME/models" --dry-run

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") <source_folder> [dest_folder] [extra args...]

  source_folder  Absolute or relative path to local folder containing .pth files
  dest_folder    Destination directory to copy files into (defaults to repo_root/downloaded_checkpoints)
  extra args     Any extra flags passed to file_downloader.py (e.g. --dry-run)

Example:
  $(basename "$0") "/home/user/checkpoints" "$HOME/models" --dry-run
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

SOURCE="$1"
shift || true

DEST=""
# If second arg is given and doesn't start with --, treat as dest
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then
  DEST="$1"
  shift || true
fi

EXTRA_ARGS=("$@")

# Resolve script directory (repo-relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [ -z "$DEST" ]; then
  # default destination relative to repo: two levels up from SCRIPT_DIR
  DEST="$SCRIPT_DIR/../../downloaded_checkpoints"
fi

# Normalize paths
SOURCE="$(readlink -f "$SOURCE")"
DEST="$(readlink -f "$DEST")"

if [ ! -d "$SOURCE" ]; then
  echo "ERROR: source folder not found: $SOURCE" >&2
  exit 2
fi

mkdir -p "$DEST"

# Prefer python3, fallback to python
PYTHON="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON" ]; then
  echo "ERROR: python is not available in PATH" >&2
  exit 3
fi

echo "Downloading .pth files from '$SOURCE' to '$DEST' ${EXTRA_ARGS[*]:-}" 

# Call the downloader script located next to this script
"$PYTHON" "$SCRIPT_DIR/file_downloader.py" --provider local --folder "$SOURCE" --dest "$DEST" --pattern "*.pth" "${EXTRA_ARGS[@]}"
RC=$?
if [ $RC -eq 0 ]; then
  echo "Download completed successfully."
else
  echo "Download failed with exit code $RC." >&2
fi

exit $RC
