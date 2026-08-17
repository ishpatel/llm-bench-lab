# Shim so `pip install -e .` works everywhere. Metadata is in setup.cfg and
# there is deliberately NO pyproject.toml: with one present, old pips (21.x,
# the macOS stock toolchain) build with the newest setuptools, whose develop
# command re-invokes pip with flags old pip lacks, while pinning setuptools<64
# instead breaks new pips that require the PEP 660 hook. The plain legacy
# layout is the one arrangement every pip since 2015 installs correctly.
# Runtime dependencies remain zero; installing is optional sugar.
from setuptools import setup

setup()
