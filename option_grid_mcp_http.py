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
"""
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
    jump_pct: Optional[float] = None,
    n_spots: int = 41,
    n_texp: int = 21,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = "pct",
    include_per_option: bool = False,
) -> Dict[str, Any]:
    #""Write an .xlsx with one sheet per field; return download_url + meta.""
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, jump_pct=jump_pct, n_spots=n_spots, n_texp=n_texp,
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

        meta = pd.DataFrame([req.model_dump()])
        meta.to_excel(writer, sheet_name="meta", index=False)

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
"""

"""
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
    jump_pct: Optional[float] = None,
    n_spots: int = 41,
    n_texp: int = 21,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = "pct",
    include_per_option: bool = False,
) -> Dict[str, Any]:
    #""Write an .xlsx with one sheet per field; return download_url + meta + extrema summary.""
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, jump_pct=jump_pct, n_spots=n_spots, n_texp=n_texp,
        spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
        axes_mode=axes_mode, include_per_option=include_per_option,
    )
    grid = option_grid(req)

    axes = grid["axes"]
    Sg = np.array(axes["S"], dtype=float)   # (m, n)
    Tg = np.array(axes["T"], dtype=float)   # (m, n)
    port = grid["portfolio_scaled"]

    def _extrema(v_like: Any):
        V = np.array(v_like, dtype=float)
        imax = int(np.nanargmax(V))
        imin = int(np.nanargmin(V))
        i_max, j_max = np.unravel_index(imax, V.shape)
        i_min, j_min = np.unravel_index(imin, V.shape)
        return {
            "max": {
                "value": float(V[i_max, j_max]),
                #"S": float(Sg[i_max, j_max]),
                #"T": float(Tg[i_max, j_max]),
                #"index": [int(i_max), int(j_max)],
            },
            "min": {
                "value": float(V[i_min, j_min]),
                #"S": float(Sg[i_min, j_min]),
                #"T": float(Tg[i_min, j_min]),
                #"index": [int(i_min), int(j_min)],
            },
        }

    # Build the summary dict safely (only if fields exist)
    summary: Dict[str, Any] = {"jump_pct_used": req.jump_pct}
    rows_summary: List[Dict[str, Any]] = []

    if "gamma_dollars_per_1pct" in port:
        ext = _extrema(port["gamma_dollars_per_1pct"])
        summary["dollar_gamma_per_1pct"] = ext
        rows_summary += [
            {"metric": "dollar_gamma_per_1pct", "kind": "max_pos", **ext["max"]},
            {"metric": "dollar_gamma_per_1pct", "kind": "max_neg", **ext["min"]},
        ]

    if "vega_per_volpt" in port:
        ext = _extrema(port["vega_per_volpt"])
        summary["vega_per_volpt"] = ext
        rows_summary += [
            {"metric": "vega_per_volpt", "kind": "max_pos", **ext["max"]},
            {"metric": "vega_per_volpt", "kind": "max_neg", **ext["min"]},
        ]

    if "jump_pnl_dn" in port:
        V = np.array(port["jump_pnl_dn"], dtype=float)
        imin = int(np.nanargmin(V))
        i_min, j_min = np.unravel_index(imin, V.shape)
        jp = {
            "value": float(V[i_min, j_min]),
            "S": float(Sg[i_min, j_min]),
            "T": float(Tg[i_min, j_min]),
            "index": [int(i_min), int(j_min)],
            "jump_pct_used": req.jump_pct,
        }
        summary["jump_pnl_dn_most_negative"] = jp
        rows_summary.append({"metric": "jump_pnl_dn", "kind": "most_negative", **jp})

    ts = int(time.time())
    fn = EXPORT_DIR / f"grid_all_{ts}.xlsx"

    def _write_book(writer):
        sheets: List[str] = []
        total = 0

        # meta & axes
        pd.DataFrame([req.model_dump()]).to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame({"S": Sg[0, :]}).to_excel(writer, sheet_name="axes_S", index=False)
        pd.DataFrame({"T": Tg[:, 0]}).to_excel(writer, sheet_name="axes_T", index=False)

        # portfolio sheets
        for field, mat in port.items():
            V = np.array(mat, dtype=float)
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

            sheets.append(sheet)
            total += int(V.size)

        # summary sheet (nice for eyeballs)
        if rows_summary:
            pd.DataFrame(rows_summary).to_excel(writer, sheet_name="summary", index=False)
            sheets.append("summary")

        return sheets, total

    try:
        with pd.ExcelWriter(fn, engine="xlsxwriter") as w:
            sheets, total_cells = _write_book(w)
    except ModuleNotFoundError:
        with pd.ExcelWriter(fn, engine="openpyxl") as w:
            sheets, total_cells = _write_book(w)

    return {
        "download_url": f"{PUBLIC_BASE_URL}/files/{fn.name}",
        "filename": fn.name,
        "fields": list(port.keys()),
        "sheets": sheets,
        "rows": total_cells,
        "summary": summary,
        # handy units breadcrumb for the LLM:
        "units": {
            "gamma_dollars_per_1pct": "gamma exposure per 1% spot move",
            "vega_per_volpt": "vega exposure per 1 vol point",
            "jump_pnl_dn": "delta-neutral jump P&L (uses jump_pct)",
        },
    }
"""

# --- helpers (live OUTSIDE the MCP tool) -------------------------------------
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def _nan_minmax_vals(v_like: Any) -> Dict[str, Optional[float]]:
    """Value-only min/max, NaN-safe. Returns {'min': float|None, 'max': float|None}."""
    V = np.asarray(v_like, dtype=float)
    if V.size == 0 or not np.isfinite(V).any():
        return {"min": None, "max": None}
    with np.errstate(all="ignore"):
        vmin = np.nanmin(V)
        vmax = np.nanmax(V)
    return {
        "min": float(vmin) if np.isfinite(vmin) else None,
        "max": float(vmax) if np.isfinite(vmax) else None,
    }

def summarize_extrema_values_only(
    port: Dict[str, Any],
    jump_pct_used: Optional[float],
    want: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a small JSON summary with value-only extrema.
    Keys you likely care about exist in `port`: 
      'gamma_dollars_per_1pct', 'vega_per_volpt', 'jump_pnl_dn'
    """
    want = want or ["gamma_dollars_per_1pct", "vega_per_volpt", "jump_pnl_dn"]
    out: Dict[str, Any] = {"jump_pct_used": jump_pct_used}

    if "gamma_dollars_per_1pct" in port and "gamma_dollars_per_1pct" in want:
        out["dollar_gamma_per_1pct"] = _nan_minmax_vals(port["gamma_dollars_per_1pct"])

    if "vega_per_volpt" in port and "vega_per_volpt" in want:
        out["vega_per_volpt"] = _nan_minmax_vals(port["vega_per_volpt"])

    if "jump_pnl_dn" in port and "jump_pnl_dn" in want:
        mm = _nan_minmax_vals(port["jump_pnl_dn"])
        out["jump_pnl_dn_most_negative"] = {"value": mm["min"], "jump_pct_used": jump_pct_used}

    return out

def bullets_from_summary(summary: Dict[str, Any]) -> List[str]:
    """Tiny, friendly bullets for the LLM/humans."""
    b: List[str] = []
    dg = summary.get("dollar_gamma_per_1pct")
    vg = summary.get("vega_per_volpt")
    jp = summary.get("jump_pnl_dn_most_negative")
    if dg and (dg.get("min") is not None or dg.get("max") is not None):
        b.append(f"dollar_gamma_per_1pct range: {dg.get('min')} ↔ {dg.get('max')}")
    if vg and (vg.get("min") is not None or vg.get("max") is not None):
        b.append(f"vega_per_volpt range: {vg.get('min')} ↔ {vg.get('max')}")
    if isinstance(jp, dict) and jp.get("value") is not None:
        j = jp.get("value")
        jpct = summary.get("jump_pct_used")
        b.append(f"most negative jump_pnl_dn: {j}" + (f" (jump_pct≈{jpct})" if jpct is not None else ""))
    return b

def rows_for_summary_sheet(summary: Dict[str, Any]) -> pd.DataFrame:
    """Optional: rows for an Excel 'summary' tab (value-only)."""
    rows: List[Dict[str, Any]] = []
    if "dollar_gamma_per_1pct" in summary:
        mm = summary["dollar_gamma_per_1pct"]
        rows += [
            {"metric": "dollar_gamma_per_1pct", "kind": "max_pos", "value": mm.get("max")},
            {"metric": "dollar_gamma_per_1pct", "kind": "max_neg", "value": mm.get("min")},
        ]
    if "vega_per_volpt" in summary:
        mm = summary["vega_per_volpt"]
        rows += [
            {"metric": "vega_per_volpt", "kind": "max_pos", "value": mm.get("max")},
            {"metric": "vega_per_volpt", "kind": "max_neg", "value": mm.get("min")},
        ]
    if "jump_pnl_dn_most_negative" in summary:
        v = summary["jump_pnl_dn_most_negative"].get("value")
        rows.append({"metric": "jump_pnl_dn", "kind": "most_negative", "value": v})
    return pd.DataFrame(rows)


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
    jump_pct: Optional[float] = None,
    n_spots: int = 41,
    n_texp: int = 21,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = "pct",
    include_per_option: bool = False,
) -> Dict[str, Any]:
    """Write an .xlsx with one sheet per field; return download_url + filename + fields + sheets + rows + summary + bullets."""
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, jump_pct=jump_pct, n_spots=n_spots, n_texp=n_texp,
        spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
        axes_mode=axes_mode, include_per_option=include_per_option,
    )
    grid = option_grid(req)

    axes = grid["axes"]
    Sg = np.array(axes["S"], dtype=float)   # (m, n)
    Tg = np.array(axes["T"], dtype=float)   # (m, n)
    port = grid["portfolio_scaled"]

    # === value-only extrema + bullets (from helpers) ===
    summary = summarize_extrema_values_only(port, jump_pct_used=req.jump_pct)
    bullets = bullets_from_summary(summary)

    ts = int(time.time())
    fn = EXPORT_DIR / f"grid_all_{ts}.xlsx"

    def _write_book(writer):
        sheets: List[str] = []
        total = 0

        # meta & axes
        pd.DataFrame([req.model_dump()]).to_excel(writer, sheet_name="meta", index=False)
        pd.DataFrame({"S": Sg[0, :]}).to_excel(writer, sheet_name="axes_S", index=False)
        pd.DataFrame({"T": Tg[:, 0]}).to_excel(writer, sheet_name="axes_T", index=False)

        # portfolio sheets
        for field, mat in port.items():
            V = np.array(mat, dtype=float)
            if V.shape != Sg.shape:
                continue
            df = pd.DataFrame(V, index=Tg[:, 0], columns=Sg[0, :])
            df.index.name = "T"; df.columns.name = "S"
            sheet = _sanitize_sheet(field)
            df.to_excel(writer, sheet_name=sheet)
            try:
                ws = writer.sheets[sheet]
                ws.freeze_panes(1, 1)
                ws.set_column(0, 0, 12)
                ws.set_column(1, df.shape[1], 12)
            except Exception:
                pass
            sheets.append(sheet); total += int(V.size)

        # optional: value-only summary tab for humans
        df_sum = rows_for_summary_sheet(summary)
        if not df_sum.empty:
            df_sum.to_excel(writer, sheet_name="summary", index=False)
            sheets.append("summary")

        # optional: bullets tab (pure text)
        if bullets:
            pd.DataFrame({"bullet": bullets}).to_excel(writer, sheet_name="bullets", index=False)
            sheets.append("bullets")

        return sheets, total

    try:
        with pd.ExcelWriter(fn, engine="xlsxwriter") as w:
            sheets, total_cells = _write_book(w)
    except ModuleNotFoundError:
        with pd.ExcelWriter(fn, engine="openpyxl") as w:
            sheets, total_cells = _write_book(w)

    return {
        "download_url": f"{PUBLIC_BASE_URL}/files/{fn.name}",
        "filename": fn.name,
        "fields": list(port.keys()),
        "sheets": sheets,
        "rows": total_cells,
        "summary": summary,         # value-only numbers for LLMs
        "bullets": bullets,         # snack-sized commentary prompts
        "units": {
            "gamma_dollars_per_1pct": "gamma exposure per 1% spot move",
            "vega_per_volpt": "vega exposure per 1 vol point",
            "jump_pnl_dn": "delta-neutral jump P&L (uses jump_pct)",
        },
    }


# 1) Make the MCP server’s HTTP path be its own root
mcp.settings.streamable_http_path = "/"   # <-- key line

# 2) Build the MCP sub-app and disable slash redirects inside it
mcp_app = mcp.streamable_http_app()
mcp_app.router.redirect_slashes = False   # accept / and // variations, no 307s

# 3) Parent app, mount at /mcp so both /mcp and /mcp/ hit mcp_app
app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com","https://chatgpt.com"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["content-type","mcp-protocol-version","mcp-session-id"],
    expose_headers=["mcp-session-id"],
)
app.mount("/mcp", mcp_app)                       # <-- both /mcp and /mcp/ work now
app.mount("/files", StaticFiles(directory=str(EXPORT_DIR)), name="files")
"""
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
"""
# tiny health
@app.route("/")
async def _health(_):
    return JSONResponse({"ok": True, "mcp": "/mcp", "files": "/files"})
