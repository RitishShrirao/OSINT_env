from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
