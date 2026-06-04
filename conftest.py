import sys
from pathlib import Path

# Ensure the repo root is importable so `import superres` and `import tests.*` work
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
