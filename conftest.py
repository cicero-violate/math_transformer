import sys
from pathlib import Path

# Allow `from src.xxx import ...` when pytest is run from this directory
sys.path.insert(0, str(Path(__file__).parent))
