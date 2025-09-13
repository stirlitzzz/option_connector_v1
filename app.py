from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any
import numpy as np


# --- MCP glue ---------------------------------------------------------------
from mcp.server.fastmcp import FastMCP
from starlette.middleware.cors import CORSMiddleware
import base64, json
from fastapi.responses import RedirectResponse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("mcp").setLevel(logging.DEBUG)


app = FastAPI()

@app.get("/health")
def health(): return {"ok": True}


@app.get("/")
def home(): return {"hello hello": "deltadisco.party"}







# Create the MCP server
mcp = FastMCP(
    name="OptionGrid",
    instructions=(
        "Compute Black–Scholes–Merton option price/greeks grids. "
        "Use `search` by passing JSON parameters in the query; it returns an ID. "
        "Then call `fetch(id)` to retrieve the computed grid."
    ),
    streamable_http_path="/"  # so the server mounts exactly at /mcp
)  # defaults to streamable HTTP; better for ChatGPT over the web


# Initialize FastAPI with a lifespan that starts FastMCP's session manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Ensure the Streamable HTTP manager task group is running
    async with mcp.session_manager.run():
        yield
mcp_app = mcp.streamable_http_app()
app = FastAPI(lifespan=lifespan)
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
    REQUIRED by ChatGPT Deep Research.
    Fetch the item by ID. If the ID is 'help:option_grid', return usage instructions.
    If the ID encodes params (produced by `search`), compute and return the grid.
    """
    if id == "help:option_grid_mcp":
        return {
            "title": "OptionGrid MCP usage",
            "how_to": "Call `search` with a JSON payload (see example in docstring). "
                      "Then call `fetch(id)` with the returned id to get the result."
        }
    if id.startswith("option_grid_mcp:"):
        params = _decode(id.split(":", 1)[1])
        return option_grid_mcp(**params)
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

def bsm_grid(Sg, strikes, cp, Tg, r, q, sigma):
    """Return per-option raw surfaces (k,m,n) for price & greeks."""
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

def scale_axes(Sg, Tg, spot0, mode='pct', days_per_year=252):
    if mode == 'pct':
        Xg = Sg / spot0 - 1.0
    elif mode == 'logm':
        Xg = np.log(Sg / spot0)
    else:
        raise ValueError("axes_mode must be 'pct' or 'logm'")
    Yg_days = Tg * days_per_year
    return Xg, Yg_days

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
    axes_mode: Optional[Literal['pct','logm']] = None
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


@mcp.tool()
def search(query: str) -> dict:
    # 1) Try strict JSON first
    try:
        j = query[query.index("{"): query.rindex("}")+1]
        params = json.loads(j)
        return {"ids": [f"option_grid:{_encode(params)}"]}
    except Exception:
        pass

    # 2) NL -> params (very simple heuristics; improve as needed)
    import re
    def find(rx, default=None, cast=float):
        m = re.search(rx, query, re.I)
        return cast(m.group(1)) if m else default

    spot = find(r"\bspot\s*[:=]?\s*([0-9.]+)", default=100.0)
    ttm  = find(r"\b(ttm|tenor|t)\s*[:=]?\s*([0-9.]+)", default=0.25)
    r    = find(r"\br\s*[:=]?\s*([0-9.]+)", default=0.05)
    q    = find(r"\bq\s*[:=]?\s*([0-9.]+)", default=0.00)

    # Lists like: strikes 100,105   sigma 0.2,0.18   cp put,call   qty -1,1
    def list_of(rx, default):
        m = re.search(rx, query, re.I)
        if not m: return default
        return [x.strip() for x in re.split(r"[ ,]+", m.group(1).strip()) if x.strip()]

    strikes = [float(x) for x in list_of(r"\bstrikes?\s*([0-9., ]+)", [str(spot)])]
    sigma   = [float(x) for x in list_of(r"\bsigma\s*([0-9., ]+)", ["0.2"])]
    cp      = [x.lower()[0] for x in list_of(r"\bcp|calls?/?puts?\s*([a-z ,]+)", ["c"])]
    qty     = [float(x) for x in list_of(r"\bqty\s*([0-9., -]+)", ["1"])]

    # Pad/trim lists to same length
    k = max(len(strikes), len(sigma), len(cp), len(qty))
    def fit(xs, fill):
        xs = xs[:k] + [fill] * max(0, k - len(xs))
        return xs
    strikes = fit(strikes, spot)
    sigma   = fit(sigma,   0.2)
    cp      = fit(cp,      "c")
    qty     = fit(qty,     1.0)

    params = {
        "spot": spot, "strikes": strikes, "cp": ["call" if x=="c" else "put" for x in cp],
        "sigma": sigma, "qty": qty, "ttm": ttm, "r": r, "q": q,
        "n_spots": 21, "n_texp": 11, "spot_lo": 0.5, "spot_hi": 1.5,
        "contract_size": 1, "axes_mode": None, "include_per_option": False
    }
    return {"ids": [f"option_grid:{_encode(params)}"]}