from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_APP_PATH = ROOT_DIR / "backend" / "app.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

spec = importlib.util.spec_from_file_location("ai_image_forensics_root_app", ROOT_APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load application module from {ROOT_APP_PATH}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.app
