from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import uvicorn


_ROOT_SERVER_PATH = Path(__file__).resolve().parents[1] / "server.py"
_SPEC = importlib.util.spec_from_file_location("osint_root_server", _ROOT_SERVER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load server module from {_ROOT_SERVER_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
app = _MODULE.app


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
