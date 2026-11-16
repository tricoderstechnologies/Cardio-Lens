"""Export the executed project outputs to website/assets/data.js (static fallback). Run from project root."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.site_data import build_payload
data = build_payload()
(ROOT / "website/assets/data.js").write_text("window.CARDIOLENS=" + json.dumps(data, separators=(",", ":")) + ";\n")
print("ok", len(data["examples"]), "examples")
