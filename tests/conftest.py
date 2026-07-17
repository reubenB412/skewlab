"""Make the repo root importable so `import skewlab` works without an editable install."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SKEWLAB_DEMO", "1")     # tests never touch the private pipeline
