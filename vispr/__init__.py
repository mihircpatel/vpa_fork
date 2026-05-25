"""Project-wide configuration.

DS_ROOT and CAFFE_ROOT can be configured via environment variables
`VISPR_DS_ROOT` and `VISPR_CAFFE_ROOT`. If not set, sensible defaults
are used (original hard-coded paths from the repository).
"""
import os

DS_ROOT = os.environ.get('VISPR_DS_ROOT', 'C:/Users/mihir/Documents/PhD/Research/Dataset/VISPR/')
# Path to caffe installation (kept for backwards compatibility)
CAFFE_ROOT = os.environ.get('VISPR_CAFFE_ROOT', '/BS/orekondy/work/opt/caffe-new/')

__all__ = ['DS_ROOT', 'CAFFE_ROOT']
