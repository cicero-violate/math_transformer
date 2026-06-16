#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v25_01_distillation import query_teacher_main


if __name__ == "__main__":
    raise SystemExit(query_teacher_main())
