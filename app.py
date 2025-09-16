from tkinter import W
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any
import numpy as np
import pandas as pd


# --- MCP glue ---------------------------------------------------------------
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware
import base64, json
from fastapi.responses import RedirectResponse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("mcp").setLevel(logging.DEBUG)
import csv, time
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import os
import re
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(exist_ok=True)          # ensure exists before mount

# serve: https://<host>/files/<filename>
app = FastAPI()

os.makedirs("exports", exist_ok=True)

app.mount("/files", StaticFiles(directory=str(EXPORT_DIR)), name="files")
@app.get("/health")
def health(): return {"ok": True}


@app.get("/")
def home(): return {"hello hello": "deltadisco.party"}

@app.get("/debug/files", include_in_schema=True)
def debug_files():
    files = sorted(p.name for p in EXPORT_DIR.glob("*"))
    return {"export_dir": str(EXPORT_DIR), "exists": EXPORT_DIR.exists(), "files": files}




mcp = FastMCP(
    name="OptionGrid",
    instructions=(
        "all tools are READ-ONLY / PURE COMPUTATION. No side effects.\n"
        "You expose TWO compute tools:\n"
        "1) `option_grid_mcp` — use when you need a grid of values over spot/time; "
        "   requires numeric fields: spot, strikes, cp, sigma, qty, ttm, r, q "
        "(plus optional grid args n_spots, n_texp, spot_lo, spot_hi, axes_mode).\n"
        "2) `option_strategy_price_mcp` — use to price an explicit option strategy "
        "(e.g., straddle/strangle/collar/spreads) given per-leg strikes/cp/sigma/qty "
        "and a SINGLE maturity ttm, r, q.\n\n"
        "Prefer calling these tools directly when you have numeric fields. "
        "Alternatively, call `search` with structured args or natural language to get an ID, "
        "then `fetch(id)`.\n"
        "Rules: `cp` must be 'call' or 'put'; ttm is years (e.g., 0.25=3m, 1.0=1y). "
        "Return the tool JSON as-is."
    ),
    streamable_http_path="/"
)


# Initialize FastAPI with a lifespan that starts FastMCP's session manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Ensure the Streamable HTTP manager task group is running
    async with mcp.session_manager.run():
        yield
mcp_app = mcp.streamable_http_app()

@app.get("/", include_in_schema=False)
def home():
    return {
        "message": "Hi! Use POST /option_grid",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }

@app.get("/mcp", include_in_schema=False)
def mcp_noslash():
    return RedirectResponse(url="/mcp/", status_code=308)  # 308 preserves method on POST




# The compute tool (typed just like your Pydantic request model)
@mcp.tool()
def option_grid_mcp(
    spot: float,
    strikes: list[float],
    cp: list[str],
    sigma: list[float],
    qty: list[float],
    ttm: float,
    r: float,
    q: float = 0.0,
    n_spots: int = 21,
    n_texp: int = 11,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: str | None = None,
    include_per_option: bool = False,
) -> dict:
    """
    READ-ONLY / PURE COMPUTATION. No side effects.
    Compute option price/greeks surfaces over a spot×time grid.
    Returns both raw and desk-style scaled portfolio surfaces.
    """
    req = OptionGridReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, n_spots=n_spots, n_texp=n_texp,
        spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
        axes_mode=axes_mode, include_per_option=include_per_option
    )
    return option_grid(req)  # reuse your FastAPI handler

# -- Helpers to encode params inside an ID for Deep Research flow --
def _encode(params: dict) -> str:
    b = json.dumps(params, separators=(",", ":"), ensure_ascii=False).encode()
    return base64.urlsafe_b64encode(b).decode().rstrip("=")

def _decode(s: str) -> dict:
    pad = "=" * (-len(s) % 4)
    return json.loads(base64.urlsafe_b64decode(s + pad).decode())

@mcp.tool()
def fetch(id: str) -> dict:
    """
    REQUIRED by Deep Research-style flows.
    Fetch a previously prepared computation by ID.

    Special help IDs:
    - 'help:option_grid_mcp'            → usage + example for grid
    - 'help:option_strategy_price_mcp'  → usage + example for strategy pricing
    """
    if id == "help:option_grid_mcp":
        return {
            "title": "OptionGrid usage",
            "how_to": (
                "Call `option_grid_mcp` directly when you have numeric fields "
                "(spot, strikes, cp, sigma, qty, ttm, r, q). "
                "Or call `search(tool=\"grid\", ...)` and then `fetch(id)`."
            ),
            "example_direct_call": {
                "name": "option_grid_mcp",
                "arguments": {
                    "spot": 100, "strikes": [95,105], "cp": ["put","call"],
                    "sigma": [0.2,0.18], "qty": [-1,1], "ttm": 0.25, "r": 0.03, "q": 0.0
                }
            }
        }

    if id == "help:option_strategy_price_mcp":
        return {
            "title": "Option Strategy Price usage",
            "how_to": (
                "Use `option_strategy_price_mcp` to price explicit legs "
                "(straddle, strangle, collar, spreads). "
                "Provide spot, per-leg strikes/cp/sigma/qty, single ttm, r, q. "
                "Or call `search(tool=\"strategy\", ...)` and then `fetch(id)`."
            ),
            "example_direct_call": {
                "name": "option_strategy_price_mcp",
                "arguments": {
                    "spot": 100, "strikes": [95,105], "cp": ["put","call"],
                    "sigma": [0.22,0.18], "qty": [-1,1], "ttm": [1.0, 1.0],"r": 0.03, "q": 0.0
                }
            }
        }

    if id.startswith("option_grid_mcp:"):
        params = _decode(id.split(":", 1)[1])
        return option_grid_mcp(**params)

    if id.startswith("option_strategy_price_mcp:"):
        params = _decode(id.split(":", 1)[1])
        return option_strategy_price_mcp(**params)

    raise ValueError(f"Unknown id: {id}")

# Mount the MCP server under /mcp (HTTP transport)
# NOTE: Streamable HTTP default path is /mcp; since we set streamable_http_path="/",
# the final URL is exactly /mcp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id", "MCP-Session-Id"],  # case-insensitive, be generous
)
app.mount("/mcp", mcp_app)
# ---------------------------------------------------------------------------

# ---------- math helpers (pure NumPy; no SciPy needed) ----------
def _norm_pdf(x): return np.exp(-0.5*x*x) / np.sqrt(2*np.pi)
def _norm_cdf(x):
    # Abramowitz–Stegun erf approximation, good to ~1e-7
    z = x / np.sqrt(2.0); s = np.sign(z); a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    erf_approx = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * np.exp(-a*a)
    return 0.5 * (1.0 + s * erf_approx)

def _flags(cp):
    cp = np.atleast_1d(cp).astype(str)
    cp = np.char.lower(cp)
    is_c = np.char.startswith(cp,'c'); is_p = np.char.startswith(cp,'p')
    if not np.all(is_c | is_p):
        bad = cp[~(is_c | is_p)]
        raise ValueError(f"invalid cp values: {bad}")
    return np.where(is_c, 'c', 'p')

def create_grid(spot_start, spot_end, t_start, t_end, n_spots, n_texp, indexing='ij'):
    S = np.linspace(spot_start, spot_end, n_spots)
    T = np.linspace(t_start,   t_end,   n_texp)
    Sg, Tg = np.meshgrid(S, T)#, indexing=indexing)   # shapes (n_spots, n_texp)
    return Sg, Tg


"""

def bsm_grid(Sg, strikes, cp, Tg, r, q, sigma):
    K   = np.asarray(strikes, float)[:, None, None]  # (k,1,1)
    sig = np.asarray(sigma,   float)[:, None, None]
    flg = _flags(cp)[:, None, None]                  # 'c'/'p' (k,1,1)

    S = Sg[None, ...]; T = Tg[None, ...]
    R = float(r); Q = float(q)

    df_r = np.exp(-R*T); df_q = np.exp(-Q*T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    sigp  = np.maximum(sig, 1e-12)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S/K) + (R - Q + 0.5*sigp**2) * T) / (sigp * sqrtT)
        d2 = d1 - sigp * sqrtT

    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    is_call = (flg == 'c')

    call = S*df_q*Nd1 - K*df_r*Nd2
    put  = K*df_r*_norm_cdf(-d2) - S*df_q*_norm_cdf(-d1)
    price = np.where(is_call, call, put)

    delta = np.where(is_call, df_q*Nd1, df_q*(Nd1 - 1.0))
    gamma = df_q * nd1 / (S * sigp * sqrtT)
    vega  = S * df_q * nd1 * sqrtT

    theta_common = -(S*df_q*nd1*sigp)/(2.0*sqrtT)
    theta = np.where(is_call,
                     theta_common + Q*S*df_q*Nd1 - R*K*df_r*Nd2,
                     theta_common - Q*S*df_q*_norm_cdf(-d1) + R*K*df_r*_norm_cdf(-d2))
    rho = np.where(is_call, K*T*df_r*Nd2, -K*T*df_r*_norm_cdf(-d2))

    # boundaries
    intrinsic = np.where(is_call, np.maximum(S-K,0.0), np.maximum(K-S,0.0))
    fwd_intr  = np.where(is_call, np.maximum(S*df_q - K*df_r, 0.0),
                                   np.maximum(K*df_r - S*df_q, 0.0))
    t0  = (T <= 1e-14); s0 = (sig <= 1e-14)
    price = np.where(t0, intrinsic, price)
    price = np.where(~t0 & s0, fwd_intr, price)
    delta = np.where(t0, np.where(is_call, (S>K).astype(float), -(S<K).astype(float)), delta)
    gamma = np.where(t0 | s0, 0.0, gamma)
    vega  = np.where(t0 | s0, 0.0, vega)
    theta = np.where(t0, 0.0, theta)
    rho   = np.where(t0, 0.0, rho)

    return {"price":price, "delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

"""



def bsm_grid(Sg, strikes, cp, Tg, r, q, sigma, jump_pct=None):
    """Return per-option raw surfaces (k,m,n) for price & greeks.
    
    If jump_pct is provided (scalar or array broadcastable to Sg), also return:
      - 'jump_pnl': price(S*(1+jump_pct)) - price(S)  (per-option PnL for a long 1)
    Jump is instantaneous: T, sigma, r, q unchanged; reprice is sticky-strike.
    """
    K   = np.asarray(strikes, float)[:, None, None]  # (k,1,1)
    sig = np.asarray(sigma,   float)[:, None, None]
    flg = _flags(cp)[:, None, None]                  # 'c'/'p' (k,1,1)

    S = Sg[None, ...]; T = Tg[None, ...]
    R = float(r); Q = float(q)

    df_r = np.exp(-R*T); df_q = np.exp(-Q*T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    sigp  = np.maximum(sig, 1e-12)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S/K) + (R - Q + 0.5*sigp**2) * T) / (sigp * sqrtT)
        d2 = d1 - sigp * sqrtT

    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    is_call = (flg == 'c')

    call = S*df_q*Nd1 - K*df_r*Nd2
    put  = K*df_r*_norm_cdf(-d2) - S*df_q*_norm_cdf(-d1)
    price = np.where(is_call, call, put)

    delta = np.where(is_call, df_q*Nd1, df_q*(Nd1 - 1.0))
    gamma = df_q * nd1 / (S * sigp * sqrtT)
    vega  = S * df_q * nd1 * sqrtT

    theta_common = -(S*df_q*nd1*sigp)/(2.0*sqrtT)
    theta = np.where(is_call,
                     theta_common + Q*S*df_q*Nd1 - R*K*df_r*Nd2,
                     theta_common - Q*S*df_q*_norm_cdf(-d1) + R*K*df_r*_norm_cdf(-d2))
    rho = np.where(is_call, K*T*df_r*Nd2, -K*T*df_r*_norm_cdf(-d2))

    # boundaries
    intrinsic = np.where(is_call, np.maximum(S-K,0.0), np.maximum(K-S,0.0))
    fwd_intr  = np.where(is_call, np.maximum(S*df_q - K*df_r, 0.0),
                                   np.maximum(K*df_r - S*df_q, 0.0))
    t0  = (T <= 1e-14); s0 = (sig <= 1e-14)
    price = np.where(t0, intrinsic, price)
    price = np.where(~t0 & s0, fwd_intr, price)
    delta = np.where(t0, np.where(is_call, (S>K).astype(float), -(S<K).astype(float)), delta)
    gamma = np.where(t0 | s0, 0.0, gamma)
    vega  = np.where(t0 | s0, 0.0, vega)
    theta = np.where(t0, 0.0, theta)
    rho   = np.where(t0, 0.0, rho)

    out = {"price":price, "delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

    # Optional jump PnL
    if jump_pct is not None:
        jp = np.asarray(jump_pct, float)  # scalar or broadcastable to Sg
        S_jump = np.maximum(Sg * (1.0 + jp), 1e-300)  # keep positive
        Sj = S_jump[None, ...]  # (1,m,n)

        with np.errstate(divide='ignore', invalid='ignore'):
            d1j = (np.log(Sj/K) + (R - Q + 0.5*sigp**2) * T) / (sigp * sqrtT)
            d2j = d1j - sigp * sqrtT

        Nd1j, Nd2j = _norm_cdf(d1j), _norm_cdf(d2j)
        callj = Sj*df_q*Nd1j - K*df_r*Nd2j
        putj  = K*df_r*_norm_cdf(-d2j) - Sj*df_q*_norm_cdf(-d1j)
        pricej = np.where(is_call, callj, putj)

        # Apply same boundary logic to the jumped price
        intrinsic_j = np.where(is_call, np.maximum(Sj-K,0.0), np.maximum(K-Sj,0.0))
        fwd_intr_j  = np.where(is_call, np.maximum(Sj*df_q - K*df_r, 0.0),
                                         np.maximum(K*df_r - Sj*df_q, 0.0))
        pricej = np.where(t0, intrinsic_j, pricej)
        pricej = np.where(~t0 & s0, fwd_intr_j, pricej)
        # Jump deltas already computed at pre-jump S
        dS = Sj - S                      # shape (1,m,n)
        jump_pnl = pricej - price        # unhedged
        jump_pnl_dn = jump_pnl - delta * dS  # delta-neutral


        out["jump_pnl"] = jump_pnl
        out["jump_pnl_dn"] = jump_pnl_dn

    return out


def bsm(Sg, strikes, cp, Tg, r, q, sigma, jump_pct=None):
    """Return per-option raw surfaces (k,m,n) for price & greeks.
    
    If jump_pct is provided (scalar or array broadcastable to Sg), also return:
      - 'jump_pnl': price(S*(1+jump_pct)) - price(S)  (per-option PnL for a long 1)
    Jump is instantaneous: T, sigma, r, q unchanged; reprice is sticky-strike.
    """
    K   = np.asarray(strikes, float)
    sig = np.asarray(sigma,   float)
    flg = _flags(cp)
    S = Sg
    T = Tg
    R = float(r); Q = float(q)

    df_r = np.exp(-R*T); df_q = np.exp(-Q*T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    sigp  = np.maximum(sig, 1e-12)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S/K) + (R - Q + 0.5*sigp**2) * T) / (sigp * sqrtT)
        d2 = d1 - sigp * sqrtT

    Nd1, Nd2 = _norm_cdf(d1), _norm_cdf(d2)
    nd1 = _norm_pdf(d1)
    is_call = (flg == 'c')

    call = S*df_q*Nd1 - K*df_r*Nd2
    put  = K*df_r*_norm_cdf(-d2) - S*df_q*_norm_cdf(-d1)
    price = np.where(is_call, call, put)

    delta = np.where(is_call, df_q*Nd1, df_q*(Nd1 - 1.0))
    gamma = df_q * nd1 / (S * sigp * sqrtT)
    vega  = S * df_q * nd1 * sqrtT

    theta_common = -(S*df_q*nd1*sigp)/(2.0*sqrtT)
    theta = np.where(is_call,
                     theta_common + Q*S*df_q*Nd1 - R*K*df_r*Nd2,
                     theta_common - Q*S*df_q*_norm_cdf(-d1) + R*K*df_r*_norm_cdf(-d2))
    rho = np.where(is_call, K*T*df_r*Nd2, -K*T*df_r*_norm_cdf(-d2))

    # boundaries
    intrinsic = np.where(is_call, np.maximum(S-K,0.0), np.maximum(K-S,0.0))
    fwd_intr  = np.where(is_call, np.maximum(S*df_q - K*df_r, 0.0),
                                   np.maximum(K*df_r - S*df_q, 0.0))
    t0  = (T <= 1e-14); s0 = (sig <= 1e-14)
    price = np.where(t0, intrinsic, price)
    price = np.where(~t0 & s0, fwd_intr, price)
    delta = np.where(t0, np.where(is_call, (S>K).astype(float), -(S<K).astype(float)), delta)
    gamma = np.where(t0 | s0, 0.0, gamma)
    vega  = np.where(t0 | s0, 0.0, vega)
    theta = np.where(t0, 0.0, theta)
    rho   = np.where(t0, 0.0, rho)

    out = {"price":price, "delta":delta, "gamma":gamma, "vega":vega, "theta":theta, "rho":rho}

    # Optional jump PnL
    if jump_pct is not None:
        jp = np.asarray(jump_pct, float)  # scalar or broadcastable to Sg
        S_jump = np.maximum(Sg * (1.0 + jp), 1e-300)  # keep positive
        Sj = S_jump[None, ...]  # (1,m,n)

        with np.errstate(divide='ignore', invalid='ignore'):
            d1j = (np.log(Sj/K) + (R - Q + 0.5*sigp**2) * T) / (sigp * sqrtT)
            d2j = d1j - sigp * sqrtT

        Nd1j, Nd2j = _norm_cdf(d1j), _norm_cdf(d2j)
        callj = Sj*df_q*Nd1j - K*df_r*Nd2j
        putj  = K*df_r*_norm_cdf(-d2j) - Sj*df_q*_norm_cdf(-d1j)
        pricej = np.where(is_call, callj, putj)

        # Apply same boundary logic to the jumped price
        intrinsic_j = np.where(is_call, np.maximum(Sj-K,0.0), np.maximum(K-Sj,0.0))
        fwd_intr_j  = np.where(is_call, np.maximum(Sj*df_q - K*df_r, 0.0),
                                         np.maximum(K*df_r - Sj*df_q, 0.0))
        pricej = np.where(t0, intrinsic_j, pricej)
        pricej = np.where(~t0 & s0, fwd_intr_j, pricej)
        # Jump deltas already computed at pre-jump S
        dS = Sj - S                      # shape (1,m,n)
        jump_pnl = pricej - price        # unhedged
        jump_pnl_dn = jump_pnl - delta * dS  # delta-neutral


        out["jump_pnl"] = jump_pnl
        out["jump_pnl_dn"] = jump_pnl_dn

    return out

def scale_axes(Sg, Tg, spot0, mode='pct', days_per_year=252):
    print(f"mode: {mode}")
    if mode == 'pct':
        Xg = Sg / spot0 - 1.0
    elif mode == 'logm':
        Xg = np.log(Sg / spot0)
    else:
        raise ValueError("axes_mode must be 'pct' or 'logm'")
    Yg_days = Tg * days_per_year
    return Xg, Yg_days


"""
def scale_surfaces(per_opt_raw, qty, Sg, contract_size=100):
    qty = np.asarray(qty, float)
    w = (contract_size * qty)[:, None, None]  # (k,1,1)
    scaled = {}
    scaled['price'] = per_opt_raw['price'] * w
    scaled['delta_shares']  = per_opt_raw['delta'] * w
    scaled['delta_dollars'] = per_opt_raw['delta'] * w * Sg[None, ...]
    scaled['gamma_shares']  = per_opt_raw['gamma'] * w
    scaled['gamma_dollars_per_1pct'] = per_opt_raw['gamma'] * w * (Sg[None, ...]**2) * 0.01
    scaled['vega_per_volpt'] = per_opt_raw['vega'] * w * 0.01
    scaled['theta_per_day']  = per_opt_raw['theta'] * w / 365.0
    scaled['rho_per_bp']     = per_opt_raw['rho']   * w / 10000.0
    portfolio = {k: v.sum(axis=0) for k, v in scaled.items()}  # (m,n)
    return scaled, portfolio
"""

def scale_surfaces(per_opt_raw, qty, Sg, contract_size=100):
    """Desk-style scaling by qty and local spot."""
    qty = np.asarray(qty, float)
    w = (contract_size * qty)[:, None, None]  # (k,1,1)
    scaled = {}
    scaled['price'] = per_opt_raw['price'] * w
    scaled['delta_shares']  = per_opt_raw['delta'] * w
    scaled['delta_dollars'] = per_opt_raw['delta'] * w * Sg[None, ...]
    scaled['gamma_shares']  = per_opt_raw['gamma'] * w
    scaled['gamma_dollars_per_1pct'] = per_opt_raw['gamma'] * w * (Sg[None, ...]**2) * 0.01
    scaled['vega_per_volpt'] = per_opt_raw['vega'] * w * 0.01
    scaled['theta_per_day']  = per_opt_raw['theta'] * w / 365.0
    scaled['rho_per_bp']     = per_opt_raw['rho']   * w / 10000.0
    if 'jump_pnl' in per_opt_raw:
        scaled['jump_pnl'] = per_opt_raw['jump_pnl'] * w
    if 'jump_pnl_dn' in per_opt_raw:
        scaled['jump_pnl_dn'] = per_opt_raw['jump_pnl_dn'] * w
    portfolio = {k: v.sum(axis=0) for k, v in scaled.items()}  # (m,n)
    return scaled, portfolio

def scale_surfaces_strategy(per_opt_raw, qty, Sg, contract_size=100):
    """Desk-style scaling by qty and local spot."""
    qty = np.asarray(qty, float)
    w = (contract_size * qty)
    scaled = {}
    scaled['price'] = per_opt_raw['price'] * w
    scaled['delta_shares']  = per_opt_raw['delta'] * w
    scaled['delta_dollars'] = per_opt_raw['delta'] * w * Sg
    scaled['gamma_shares']  = per_opt_raw['gamma'] * w
    scaled['gamma_dollars_per_1pct'] = per_opt_raw['gamma'] * w * (Sg**2) * 0.01
    scaled['vega_per_volpt'] = per_opt_raw['vega'] * w * 0.01
    scaled['theta_per_day']  = per_opt_raw['theta'] * w / 365.0
    scaled['rho_per_bp']     = per_opt_raw['rho']   * w / 10000.0
    if 'jump_pnl' in per_opt_raw:
        scaled['jump_pnl'] = per_opt_raw['jump_pnl'] * w
    if 'jump_pnl_dn' in per_opt_raw:
        scaled['jump_pnl_dn'] = per_opt_raw['jump_pnl_dn'] * w
    portfolio = {k: v.sum(axis=0) for k, v in scaled.items()}  # (m,n)
    return scaled, portfolio

# ---------- API models ----------
class OptionGridReq(BaseModel):
    spot: float
    strikes: List[float]
    cp: List[str]           # 'call'/'put' or 'c'/'p'
    sigma: List[float]
    qty: List[float]
    ttm: float              # years, grid goes ttm -> 0
    r: float
    q: float = 0.0
    n_spots: int = 41
    n_texp: int = 21
    spot_lo: float = 0.5
    spot_hi: float = 1.5
    contract_size: int = 100
    axes_mode: Optional[Literal['pct','logm']] = 'pct'
    include_per_option: bool = False

# ---------- endpoint ----------
def _tolist_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

@app.post("/option_grid", response_model=dict)
def option_grid(req: OptionGridReq):
    # 1) grids
    Sg, Tg = create_grid(req.spot*req.spot_lo, req.spot*req.spot_hi,
                         req.ttm, 0.0, req.n_spots, req.n_texp, indexing='ij')

    # 2) per-option raw surfaces (k,m,n)
    per_opt_raw = bsm_grid(Sg, req.strikes, req.cp, Tg, req.r, req.q, req.sigma)

    # 3) scaled per-option + portfolio
    per_opt_scaled, portfolio_scaled = scale_surfaces(per_opt_raw, req.qty, Sg, contract_size=req.contract_size)

    # 4) raw portfolio (sum over options)
    portfolio_raw = {k: v.sum(axis=0) for k, v in per_opt_raw.items()}  # (m,n)

    # 5) axes
    axes = {"S": Sg, "T": Tg}
    if req.axes_mode:
        Xg, Yg = scale_axes(Sg, Tg, req.spot, mode=req.axes_mode)
        axes.update({"X": Xg, "Y": Yg})

    resp = {
        "meta": {
            "spot0": req.spot, "r": req.r, "q": req.q, "ttm_start": req.ttm,
            "strikes": req.strikes, "cp": req.cp, "sigma": req.sigma, "qty": req.qty,
            "n_spots": req.n_spots, "n_texp": req.n_texp, "spot_lo": req.spot_lo, "spot_hi": req.spot_hi,
            "contract_size": req.contract_size
        },
        "axes": _tolist_dict(axes),
        #"portfolio_raw": _tolist_dict(portfolio_raw),           # (m,n)
        "portfolio_scaled": _tolist_dict(portfolio_scaled),     # (m,n)
    }
    """
    if req.include_per_option:
        resp["per_option_raw"]    = {k: v.tolist() for k, v in per_opt_raw.items()}       # (k,m,n)
        resp["per_option_scaled"] = {k: v.tolist() for k, v in per_opt_scaled.items()}    # (k,m,n)
    """
    return resp

# ---------- API models ----------
class OptionStrategyPriceReq(BaseModel):
    spot: float
    strikes: List[float]
    cp: List[str]           # 'call'/'put' or 'c'/'p'
    sigma: List[float]
    qty: List[float]
    ttm: List[float]              # years, grid goes ttm -> 0
    r: float
    q: float = 0.0
    contract_size: int = 100


@mcp.tool()
def option_strategy_price_mcp(
    spot: float,
    strikes: List[float],
    cp: List[str],           # 'call'/'put' or 'c'/'p'
    sigma: List[float],
    qty: List[float],
    ttm: List[float],             # years, grid goes ttm -> 0
    r: float,
    q: float = 0.0,
    contract_size: int = 100
) -> dict:
    """
    READ-ONLY / PURE COMPUTATION. No side effects.
    Use this when the user wants to **price an option strategy** (straddles, strangles, collars, spreads, etc.)
    given explicit option legs. Do NOT use this to compute full grids of values (use `option_grid_mcp` for that).

    **Arguments**
    - `spot` (float): Current underlying spot price.
    - `strikes` (list[float]): Strike price(s) of each leg.
    - `cp` (list[str]): Option type(s) for each leg. Must be `"call"` or `"put"` (or `"c"`/`"p"`).
    - `sigma` (list[float]): Volatility for each leg (annualized, e.g. 0.20 for 20%).
    - `qty` (list[float]): Quantity for each leg. Positive = long, negative = short.
    - `ttm` (float): Time to maturity in years (e.g. 0.25 = 3 months, 1.0 = 1 year).
    - `r` (float): Risk-free interest rate (annualized).
    - `q` (float, optional): Continuous dividend yield. Default = 0.0.
    - `contract_size` (int, optional): Number of underlying units per contract. Default = 100.

    **Behavior**
    - Each leg is priced via Black–Scholes–Merton given its strike, type, vol, maturity, and rate inputs.
    - Scales prices and Greeks by quantity and contract size.
    - Aggregates to produce both per-option and portfolio-level outputs.

    **Returns**
    A JSON dict with:
    - `meta`: Echo of inputs (spot, strikes, cp, sigma, qty, r, q, ttm, contract_size).
    - `per_opt_scaled`: Scaled price/Greeks for each individual option leg.
    - `portfolio_scaled`: Combined portfolio totals (sum over all legs).
    - `Sg`: List of spot values used internally (usually constant array of `spot`).
    - `Tg`: List with the maturity value(s).

    **Example Call**
    ```
    {
      "spot": 100,
      "strikes": [95, 105],
      "cp": ["put", "call"],
      "sigma": [0.22, 0.18],
      "qty": [-1, 1],
      "ttm": 1.0,
      "r": 0.03,
      "q": 0.0
    }
    ```
    """
    req = OptionStrategyPriceReq(
        spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
        ttm=ttm, r=r, q=q, contract_size=contract_size
    )
    return option_strategy_price(req)  # reuse your FastAPI handler







@app.post("/option_strategy_price", response_model=dict)
def option_strategy_price(req: OptionStrategyPriceReq):
    # 1) grids
    Sg=np.array(req.spot)*np.ones(len(req.strikes))
    Tg=np.array(req.ttm)

    # 2) per-option raw surfaces (k,m,n)
    per_opt_raw = bsm(Sg, req.strikes, req.cp, Tg, req.r, req.q, req.sigma,jump_pct=-.5)
    #per_opt_raw={k:v[:,0,0] for k,v in per_opt_raw.items()}

    # 3) scaled per-option + portfolio
    per_opt_scaled, portfolio_scaled = scale_surfaces_strategy(per_opt_raw, req.qty, Sg, contract_size=req.contract_size)
    #per_opt_scaled={k:v[:,0,0] for k,v in per_opt_scaled.items()}
    #portfolio_scaled={k:v[0,0] for k,v in portfolio_scaled.items()}


    resp = {
        "meta": {
            "spot0": req.spot, "r": req.r, "q": req.q, "ttm_start": req.ttm,
            "strikes": req.strikes, "cp": req.cp, "sigma": req.sigma, "qty": req.qty,
            "contract_size": req.contract_size
        },
        "per_opt_scaled": _tolist_dict(per_opt_scaled),          
        "portfolio_scaled": _tolist_dict(portfolio_scaled)   
        #"Tg": Tg.tolist(),
        #"Sg": Sg.tolist()
    }
    return resp




from typing import Optional, Literal

@mcp.tool()
def search(
    query: Optional[str] = None,
    # structured fields (preferred)
    tool: Optional[Literal["grid","strategy"]] = None,
    spot: Optional[float] = None,
    strikes: Optional[List[float]] = None,
    cp: Optional[List[str]] = None,        # 'call'/'put' or 'c'/'p'
    sigma: Optional[List[float]] = None,
    qty: Optional[List[float]] = None,
    ttm: Optional[float] = None,
    r: Optional[float] = None,
    q: Optional[float] = None,
    # grid-only extras (kept for compatibility)
    n_spots: int = 21,
    n_texp: int = 11,
    spot_lo: float = 0.5,
    spot_hi: float = 1.5,
    contract_size: int = 100,
    axes_mode: Optional[str] = None,
    include_per_option: bool = False,
) -> dict:
    """
    Search helper that returns an ID for `fetch(id)`. Two modes:

    1) Structured: pass numeric fields directly. Set `tool="grid"` or `tool="strategy"`.
       If `tool` is omitted, we prefer 'strategy' when ttm is a single float and legs look explicit.
    2) Natural language (query): we parse common tokens and default to 'strategy'.

    Returns: {"ids": ["<tool_name>:<base64-json>"]} where tool_name ∈
             {"option_grid_mcp", "option_strategy_price_mcp"}.
    """
    def norm_cp(xs: List[str]) -> List[str]:
        return [("call" if x.lower().startswith("c") else "put") for x in xs]

    # ---------- Structured path ----------
    have_struct = all(v is not None for v in (spot, strikes, cp, sigma, qty, ttm, r, q))
    if have_struct:
        params = {
            "spot": float(spot),
            "strikes": [float(x) for x in strikes],   # per-leg
            "cp": norm_cp(cp),
            "sigma": [float(x) for x in sigma],
            "qty": [float(x) for x in qty],
            "ttm": float(ttm),
            "r": float(r),
            "q": float(q),
            "contract_size": int(contract_size),
        }

        # Choose tool
        chosen = tool
        if chosen is None:
            # Heuristic: if caller provided grid-ish fields, assume grid; else strategy
            gridish = axes_mode is not None or n_spots != 21 or n_texp != 11 or spot_lo != 0.5 or spot_hi != 1.5 or include_per_option
            chosen = "grid" if gridish else "strategy"

        if chosen == "grid":
            params.update(dict(
                n_spots=int(n_spots), n_texp=int(n_texp),
                spot_lo=float(spot_lo), spot_hi=float(spot_hi),
                axes_mode=axes_mode, include_per_option=bool(include_per_option)
            ))
            return {"ids": [f"option_grid_mcp:{_encode(params)}"]}

        # strategy
        return {"ids": [f"option_strategy_price_mcp:{_encode(params)}"]}

    # ---------- NL fallback ----------
    if query:
        import re
        def find(rx, default=None, cast=float):
            m = re.search(rx, query, re.I)
            return cast(m.group(1)) if m else default

        s = find(r"\bspot\s*[:=]?\s*([0-9.]+)", 100.0)
        t = find(r"\b(ttm|tenor|t)\s*[:=]?\s*([0-9.]+)", 0.25)
        rr = find(r"\br\s*[:=]?\s*([0-9.]+)", 0.03)
        qq = find(r"\bq\s*[:=]?\s*([0-9.]+)", 0.00)

        def list_of(rx, default):
            m = re.search(rx, query, re.I)
            if not m: return default
            return [x.strip() for x in re.split(r"[ ,]+", m.group(1).strip()) if x.strip()]

        k_strikes = [float(x) for x in list_of(r"\bstrikes?\s*([0-9., ]+)", [str(s)])]
        k_sigma   = [float(x) for x in list_of(r"\bsigma\s*([0-9., ]+)", ["0.2"])]
        k_cp      = [x.lower()[0] for x in list_of(r"\bcp|calls?/?puts?\s*([a-z ,]+)", ["c"])]
        k_qty     = [float(x) for x in list_of(r"\bqty\s*([0-9., -]+)", ["1"])]

        L = max(len(k_strikes), len(k_sigma), len(k_cp), len(k_qty))
        def fit(xs, fill): return (xs[:L] + [fill] * max(0, L - len(xs)))
        k_strikes = fit(k_strikes, s)
        k_sigma   = fit(k_sigma,   0.2)
        k_cp      = fit(k_cp,      "c")
        k_qty     = fit(k_qty,     1.0)

        params = {
            "spot": s, "strikes": k_strikes, "cp": norm_cp(k_cp),
            "sigma": k_sigma, "qty": k_qty, "ttm": t, "r": rr, "q": qq,
            "contract_size": contract_size
        }

        # NL: default to strategy unless the user says "grid"
        if re.search(r"\bgrid\b", query, re.I):
            params.update(dict(
                n_spots=n_spots, n_texp=n_texp, spot_lo=spot_lo, spot_hi=spot_hi,
                axes_mode=axes_mode, include_per_option=include_per_option
            ))
            return {"ids": [f"option_grid_mcp:{_encode(params)}"]}

        return {"ids": [f"option_strategy_price_mcp:{_encode(params)}"]}

    # ---------- no usable input ----------
    return {"ids": ["help:option_strategy_price_mcp"]}


from pydantic import Field
from fastapi import HTTPException


class ExportGridReq(BaseModel):
    spot: float
    strikes: List[float]
    cp: List[str]                       # 'call'/'put' or 'c'/'p'
    sigma: List[float]
    qty: List[float]
    ttm: float                          # years
    r: float
    q: float = 0.0
    n_spots: int = 41
    n_texp: int = 21
    spot_lo: float = 0.5
    spot_hi: float = 1.5
    contract_size: int = 100
    axes_mode: Optional[str] = 'pct'
    include_per_option: bool = False
    field: str = Field(default="price", description="portfolio field to export")
    

@app.post("/export_option_grid_csv", include_in_schema=True)
def export_option_grid_csv(req: ExportGridReq):
    # Build and compute the grid using your existing code
    og_req = OptionGridReq(
        spot=req.spot, strikes=req.strikes, cp=req.cp, sigma=req.sigma, qty=req.qty,
        ttm=req.ttm, r=req.r, q=req.q, n_spots=req.n_spots, n_texp=req.n_texp,
        spot_lo=req.spot_lo, spot_hi=req.spot_hi, contract_size=req.contract_size,
        axes_mode=req.axes_mode, include_per_option=req.include_per_option
    )
    result = option_grid(og_req)  # <- your existing handler returns JSON

    # Pull axes + the chosen portfolio field
    axes = result["axes"]
    Sg = np.array(axes["S"])      # (m,n)
    Tg = np.array(axes["T"])      # (m,n)
    port = result["portfolio_scaled"]

    field = req.field
    if field not in port:
        raise HTTPException(status_code=400, detail={
            "error": f"field '{field}' not in portfolio_scaled",
            "available": list(port.keys())
        })

    V = np.array(port[field])     # (m,n)
    #V = pd.pivot_table(pd.DataFrame(V), index=Tg, columns=Sg, values=field)
    V_df = pd.DataFrame(V, index=Tg[:,0], columns=Sg[0,:])

    # Write tidy CSV: S,T,value (one row per grid cell)
    import csv, time
    ts = int(time.time())
    fn = EXPORT_DIR / f"grid_{field}_{ts}.csv"
    V_df.to_csv(fn)
    #with open(fn, "w", newline="") as f:
    #    w = csv.writer(f)
        #w.writerow(["S","T","value"])
    #    w.writerow(V.columns)
    #    m, n = V.shape
    #    for i in range(m):
    #        for j in range(n):
    #            w.writerow([float(Sg[i, j]), float(Tg[i, j]), float(V[i, j])])
    print(f"wrote to {fn}")

    return {
        "field": field,
        "rows": int(V.size),
        # swap domain accordingly for local vs public testing
        #"download_url": f"https://deltadisco.party/files/{fn.name}"
        # For local test use: 
        "download_url": f"http://127.0.0.1:8000/files/{fn.name}"
    }



def _sanitize_sheet(name: str) -> str:
    # Excel sheet names: max 31 chars, no []:*?/\
    name = re.sub(r'[\[\]\:\*\?\/\\]', '_', str(name))
    return (name or "sheet")[:31]

@app.post("/export_option_grid_xlsx", include_in_schema=True)
def export_option_grid_xlsx(req: ExportGridReq):
    # Build and compute the grid using your existing code
    og_req = OptionGridReq(
        spot=req.spot, strikes=req.strikes, cp=req.cp, sigma=req.sigma, qty=req.qty,
        ttm=req.ttm, r=req.r, q=req.q, n_spots=req.n_spots, n_texp=req.n_texp,
        spot_lo=req.spot_lo, spot_hi=req.spot_hi, contract_size=req.contract_size,
        axes_mode=req.axes_mode, include_per_option=req.include_per_option
    )
    result = option_grid(og_req)

    # Axes + portfolio dict
    axes = result["axes"]
    Sg = np.array(axes["S"])      # (m,n)
    Tg = np.array(axes["T"])      # (m,n)
    port = result["portfolio_scaled"]

    # Which fields to export?
    export_all = str(req.field).lower() in ("*", "all")
    if not export_all and req.field not in port:
        raise HTTPException(status_code=400, detail={
            "error": f"field '{req.field}' not in portfolio_scaled",
            "available": list(port.keys())
        })
    fields = list(port.keys()) if export_all else [req.field]

    # Build the workbook
    ts = int(time.time())
    fn = EXPORT_DIR / f"grid_{'all' if export_all else fields[0]}_{ts}.xlsx"

    def _write_workbook(writer):
        sheets_written = []
        total_cells = 0

        # optional meta sheet (handy for auditors / future-you)
        meta = pd.DataFrame([req.model_dump()])
        meta.to_excel(writer, sheet_name="meta", index=False)

        # also drop the S and T axes as separate tabs for clarity
        pd.DataFrame({"S": Sg[0, :]}).to_excel(writer, sheet_name="axes_S", index=False)
        pd.DataFrame({"T": Tg[:, 0]}).to_excel(writer, sheet_name="axes_T", index=False)

        for field in fields:
            V = np.array(port[field])  # (m,n)
            if V.shape != Sg.shape:
                # skip weird shapes (e.g., nested/per-option summaries)
                continue

            df = pd.DataFrame(V, index=Tg[:, 0], columns=Sg[0, :])
            df.index.name = "T"
            df.columns.name = "S"

            sheet = _sanitize_sheet(field)
            df.to_excel(writer, sheet_name=sheet)

            # nice-to-haves: freeze header & set simple column widths
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 1)
            ws.set_column(0, 0, 12)  # T
            ws.set_column(1, df.shape[1], 12)  # S columns

            sheets_written.append(sheet)
            total_cells += int(V.size)

        return sheets_written, total_cells

    # prefer xlsxwriter if available; fall back to openpyxl
    try:
        with pd.ExcelWriter(fn, engine="xlsxwriter") as writer:
            sheets_written, total_cells = _write_workbook(writer)
    except ModuleNotFoundError:
        with pd.ExcelWriter(fn, engine="openpyxl") as writer:
            sheets_written, total_cells = _write_workbook(writer)

    print(f"wrote workbook to {fn}")

    return {
        "fields": fields,
        "sheets": sheets_written,
        "rows": total_cells,
        # swap domain accordingly for local vs public
        # "download_url": f"https://deltadisco.party/files/{fn.name}"
        "download_url": f"http://127.0.0.1:8000/files/{fn.name}"
    }

"""
@mcp.tool()
def export_option_grid_csv_mcp(
    spot: float,
    strikes: List[float],
    cp: List[str],                 # 'call'/'put' or 'c'/'p'
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
    axes_mode: Optional[str] = None,
    include_per_option: bool = False,
    field: str = "price"           # e.g. "price","delta_shares","theta_per_day","vega_per_volpt"
) -> dict:
    #""
    Thin wrapper: validate with ExportGridReq and delegate to the FastAPI handler
    so the logic lives in one place (export_option_grid_csv).
    #""
    # Normalize to match the API’s default unless caller overrides
    axes_mode = axes_mode if axes_mode is not None else "pct"

    try:
        req = ExportGridReq(
            spot=spot, strikes=strikes, cp=cp, sigma=sigma, qty=qty,
            ttm=ttm, r=r, q=q, n_spots=n_spots, n_texp=n_texp,
            spot_lo=spot_lo, spot_hi=spot_hi, contract_size=contract_size,
            axes_mode=axes_mode, include_per_option=include_per_option,
            field=field
        )
    except Exception as e:
        return {"error": f"invalid arguments: {e}"}

    try:
        # Call the actual FastAPI route function (no HTTP hop needed)
        return export_option_grid_csv(req)
    except HTTPException as e:
        # Mirror the API's error shape for consistency
        detail = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        detail["status_code"] = getattr(e, "status_code", 400)
        return detail
    except Exception as e:
        return {"error": str(e)}
"""