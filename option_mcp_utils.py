
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any
import numpy as np

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
    


# ---------- endpoint ----------
def _tolist_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

class OptionGridReq(BaseModel):
    spot: float
    strikes: List[float]
    cp: List[str]           # 'call'/'put' or 'c'/'p'
    sigma: List[float]
    qty: List[float]
    ttm: float              # years, grid goes ttm -> 0
    r: float
    q: float = 0.0
    jump_pct: Optional[float] = None
    n_spots: int = 41
    n_texp: int = 21
    spot_lo: float = 0.5
    spot_hi: float = 1.5
    contract_size: int = 100
    axes_mode: Optional[Literal['pct','logm']] = 'pct'
    include_per_option: bool = False



def option_grid(req: OptionGridReq):
    # 1) grids
    Sg, Tg = create_grid(req.spot*req.spot_lo, req.spot*req.spot_hi,
                         req.ttm, 0.0, req.n_spots, req.n_texp, indexing='ij')

    # 2) per-option raw surfaces (k,m,n)
    per_opt_raw = bsm_grid(Sg, req.strikes, req.cp, Tg, req.r, req.q, req.sigma, req.jump_pct)

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
