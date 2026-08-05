"""Compatibility launcher for the :mod:`regime.cli` command package.

This file also permits ``python src/regime/cli.py`` during source-tree development;
installed applications should use the ``regime`` console script.
"""

from regime.cli import main

if __name__ == "__main__":
    main()
