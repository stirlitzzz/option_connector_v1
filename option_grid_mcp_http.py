# option_grid_mcp_http.py
# Requires: pip install "mcp[cli]" starlette uvicorn numpy pandas openpyxl XlsxWriter
# Run: uvicorn option_grid_mcp_http:app --host 0.0.0.0 --port 8000

from __future__ import annotations

import os, re, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# === Your existing logic (import from your app.py) ============================
from option_mcp_utils import OptionGridReq, option_grid  # make sure these names match your file

# === MCP SDK ================================================================
from mcp.server.fastmcp import FastMCP

# === Starlette wrapper: CORS + static files for downloads ====================
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", Path(__file__).parent / "exports"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# In prod set: PUBLIC_BASE_URL=https://your-domain.tld
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# MCP server + tools
# -----------------------------------------------------------------------------
mcp = FastMCP(
    name="OptionGrid",
    instructions=(
        "Tools are read-only / pure compute.\n"
        "- option_grid_mcp: compute the option grid and return JSON.\n"
        "- export_option_grid_xlsx_all: write an .xlsx with one sheet per portfolio field."
    ),
    # put the HTTP transport exactly at /mcp on this app
    streamable_http_path="/mcp",
)

@mcp.tool()
def option_grid_mcp(
    spot: float,
    strikes: List[float],
    cp: List[str],
    sigma: List[float],
    qty: List[float],
    ttm: float,
    r: float,
    q: float = 0.0,
    n_spots: int = 41,
    n_texp: int = 21,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = "pct",
    include_per_option: bool = False,
) -> Dict[str, Any]:
    """Compute the grid using your existing code and return its JSON."""
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, n_spots=n_spots, n_texp=n_texp,
        spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
        axes_mode=axes_mode, include_per_option=include_per_option,
    )
    return option_grid(req)

def _sanitize_sheet(name: str) -> str:
    name = re.sub(r'[\[\]\:\*\?\/\\]', '_', str(name))
    return (name or "sheet")[:31]

@mcp.tool()
def export_option_grid_xlsx_all(
    spot: float,
    strikes: List[float],
    cp: List[str],
    sigma: List[float],
    qty: List[float],
    ttm: float,
    r: float,
    q: float = 0.0,
    n_spots: int = 41,
    n_texp: int = 21,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = "pct",
    include_per_option: bool = False,
) -> Dict[str, Any]:
    """Write an .xlsx with one sheet per field; return download_url + meta."""
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, n_spots=n_spots, n_texp=n_texp,
        spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
        axes_mode=axes_mode, include_per_option=include_per_option,
    )
    grid = option_grid(req)

    axes = grid["axes"]
    Sg = np.array(axes["S"])      # (m, n)
    Tg = np.array(axes["T"])      # (m, n)
    port = grid["portfolio_scaled"]

    ts = int(time.time())
    fn = EXPORT_DIR / f"grid_all_{ts}.xlsx"

    def _write_book(writer):
        sheets: List[str] = []
        total = 0
        # optional helper tabs
        pd.DataFrame({"S": Sg[0, :]}).to_excel(writer, sheet_name="axes_S", index=False)
        pd.DataFrame({"T": Tg[:, 0]}).to_excel(writer, sheet_name="axes_T", index=False)

        for field, mat in port.items():
            V = np.array(mat)
            if V.shape != Sg.shape:
                continue
            df = pd.DataFrame(V, index=Tg[:, 0], columns=Sg[0, :])
            df.index.name = "T"; df.columns.name = "S"
            sheet = _sanitize_sheet(field)
            df.to_excel(writer, sheet_name=sheet)
            try:
                ws = writer.sheets[sheet]  # xlsxwriter niceties
                ws.freeze_panes(1, 1)
                ws.set_column(0, 0, 12)
                ws.set_column(1, df.shape[1], 12)
            except Exception:
                pass
            sheets.append(sheet); total += int(V.size)
        return sheets, total

    try:
        with pd.ExcelWriter(fn, engine="xlsxwriter") as w:
            sheets, total_cells = _write_book(w)
    except ModuleNotFoundError:
        with pd.ExcelWriter(fn, engine="openpyxl") as w:
            sheets, total_cells = _write_book(w)

    return {
        "download_url": f"{PUBLIC_BASE_URL}/files/{fn.name}",
        "fields": list(port.keys()),
        "sheets": sheets,
        "rows": total_cells,
    }

# -----------------------------------------------------------------------------
# Build the ASGI app straight from FastMCP (it’s a Starlette app already)
# IMPORTANT: In this “SDK flavor”, we do NOT try to start managers ourselves.
# -----------------------------------------------------------------------------
app: Starlette = mcp.streamable_http_app()  # has /mcp and /mcp/ routes inside

# niceties
app.router.redirect_slashes = False  # so /mcp doesn’t 307 to /mcp/

# CORS so ChatGPT’s browser can POST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com", "https://chatgpt.com"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type", "mcp-protocol-version", "mcp-session-id"],
    expose_headers=["mcp-session-id"],
)

# static file hosting for downloads
app.mount("/files", StaticFiles(directory=str(EXPORT_DIR)), name="files")

# tiny health
@app.route("/")
async def _health(_):
    return JSONResponse({"ok": True, "mcp": "/mcp", "files": "/files"})
