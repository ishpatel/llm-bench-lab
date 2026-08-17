"""Allow `python -m llmbench` from anywhere, as an alias for bench.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(bench.main())
