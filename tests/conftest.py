import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                       # scripts use relative paths such as models/
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
