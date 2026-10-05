"""Convenience launcher: `python main.py` (equivalent to `python -m humanslice`)."""

from humanslice.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
