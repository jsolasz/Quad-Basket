#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


@dataclass(frozen=True)
class QuadBasket:
    name: str
    longs: list[str]
    shorts: list[str]


QUAD_BASKETS: list[QuadBasket] = [
    QuadBasket(
        name="Quad 1 (Goldilocks)",
        longs=["SPY", "QQQ", "IWM", "HYG", "DBC", "FXE"],
        shorts=["TLT", "UUP"],
    ),
    QuadBasket(
        name="Quad 2 (Reflation)",
        longs=["DBC", "XLE", "XLF", "XLI", "HYG", "IWM"],
        shorts=["TLT", "IEF", "UUP"],
    ),
    QuadBasket(
        name="Quad 3 (Stagflation)",
        longs=["GLD", "DBC", "TLT", "XLU", "XLP", "XLV"],
        shorts=["LQD", "XLY", "XLF"],
    ),
    QuadBasket(
        name="Quad 4 (Deflation)",
        longs=["TLT", "IEF", "LQD", "GLD", "UUP", "XLP", "XLV"],
        shorts=["DBC", "XLE", "SPY", "QQQ", "HYG", "FXE"],
    ),
]

QUAD_COLORS = {
    "Quad 1 (Goldilocks)": "#2ca25f",
    "Quad 2 (Reflation)": "#f39c12",
    "Quad 3 (Stagflation)": "#d62728",
    "Quad 4 (Deflation)": "#d946ef",
}

QUAD_BASE_COLORS = {
    "Quad 1": "#2ca25f",
    "Quad 2": "#f39c12",
    "Quad 3": "#d62728",
    "Quad 4": "#d946ef",
}

CORE_QUAD_BASKETS = QUAD_BASKETS


# Quad playbook split into 4 toggle buckets: Core / Sector / Factor / Fixed Income.
CATEGORY_BUCKETS: dict[str, dict[str, dict[str, list[str]]]] = {
    "core": {
        "Quad 1 (Goldilocks)": {
            "longs": ["SPY", "QQQ", "IWM", "HYG", "DBC", "FXE"],
            "shorts": ["TLT", "UUP"],
        },
        "Quad 2 (Reflation)": {
            "longs": ["DBC", "XLE", "XLF", "XLI", "HYG", "IWM"],
            "shorts": ["TLT", "IEF", "UUP"],
        },
        "Quad 3 (Stagflation)": {
            "longs": ["GLD", "DBC", "TLT", "XLU", "XLP", "XLV"],
            "shorts": ["LQD", "XLY", "XLF"],
        },
        "Quad 4 (Deflation)": {
            "longs": ["TLT", "IEF", "LQD", "GLD", "UUP", "XLP", "XLV"],
            "shorts": ["DBC", "XLE", "SPY", "QQQ", "HYG", "FXE"],
        },
    },
    "sector": {
        "Quad 1 (Goldilocks)": {
            "longs": ["XLK", "XLY", "XLC", "XLI", "XLB", "VNQ"],
            "shorts": ["XLP", "XLV", "XLU"],
        },
        "Quad 2 (Reflation)": {
            "longs": ["XLE", "XLI", "XLF", "XLK", "XLY"],
            "shorts": ["XLU", "XLP", "XLV", "VNQ"],
        },
        "Quad 3 (Stagflation)": {
            "longs": ["XLU", "XLE", "VNQ", "XLP", "XLV"],
            "shorts": ["XLF", "XLC", "XLY", "XLI", "XLB"],
        },
        "Quad 4 (Deflation)": {
            "longs": ["XLP", "XLV", "XLU"],
            "shorts": ["XLE", "XLK", "XLF", "XLI", "XLY"],
        },
    },
    "factor": {
        "Quad 1 (Goldilocks)": {
            "longs": ["MTUM", "SPHB", "IWO", "IJH"],
            "shorts": ["USMV", "SPLV"],
        },
        "Quad 2 (Reflation)": {
            "longs": ["SPHB", "IWM", "MTUM", "IWO"],
            "shorts": ["USMV", "SPLV"],
        },
        "Quad 3 (Stagflation)": {
            "longs": ["MTUM", "IWO", "IJH", "QUAL"],
            "shorts": ["IWM", "DVY", "VTV"],
        },
        "Quad 4 (Deflation)": {
            "longs": ["USMV", "SPLV", "DVY", "QUAL", "VTV"],
            "shorts": ["MTUM", "SPHB"],
        },
    },
    "fixed_income": {
        "Quad 1 (Goldilocks)": {
            "longs": [],
            "shorts": [],
        },
        "Quad 2 (Reflation)": {
            "longs": [],
            "shorts": [],
        },
        "Quad 3 (Stagflation)": {
            "longs": ["TLT", "EMB", "MUB"],
            "shorts": ["LQD", "HYG"],
        },
        "Quad 4 (Deflation)": {
            "longs": ["LQD", "MUB", "BND", "AGG"],
            "shorts": ["HYG", "JNK"],
        },
    },
}


def sampled_tickvals(x_vals, max_labels: int = 12) -> list[str]:
    vals = list(x_vals)
    if not vals:
        return []
    if len(vals) <= max_labels:
        return vals
    step = int(np.ceil(len(vals) / max_labels))
    sampled = vals[::step]
    if sampled[-1] != vals[-1]:
        sampled.append(vals[-1])
    return sampled


def quad_label(name: str) -> str:
    for i in ("1", "2", "3", "4"):
        if name.startswith(f"Quad {i}"):
            return f"Quad {i}"
    return name


def quad_color(name: str) -> str:
    return QUAD_BASE_COLORS.get(quad_label(name), QUAD_COLORS.get(name, "#555555"))


def periods_per_year(interval: str) -> float:
    mapping = {
        "5m": 78.0 * 252.0,
        "15m": 26.0 * 252.0,
        "30m": 13.0 * 252.0,
        "60m": 6.5 * 252.0,
        "1d": 252.0,
    }
    return mapping.get(interval, 252.0)


def annualized_sharpe(returns: pd.Series, ppy: float) -> float:
    r = returns.dropna()
    if r.size < 2:
        return float("nan")
    vol = float(r.std(ddof=1))
    if vol <= 0:
        return float("nan")
    return float((r.mean() / vol) * math.sqrt(ppy))


def nw_default_lag(n: int) -> int:
    if n <= 1:
        return 0
    return int(max(1, math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))


def ols_hac(y: np.ndarray, x: np.ndarray, lag: int | None = None) -> dict[str, np.ndarray | float]:
    yv = np.asarray(y, dtype=float).reshape(-1, 1)
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    n = int(yv.shape[0])
    k = int(xv.shape[1])
    if n <= k + 1:
        return {
            "beta": np.full(k, np.nan),
            "se_hac": np.full(k, np.nan),
            "t_hac": np.full(k, np.nan),
            "p_two_sided_norm": np.full(k, np.nan),
            "r2": np.nan,
            "lag": float("nan"),
        }

    xtx = xv.T @ xv
    xtx_inv = np.linalg.pinv(xtx)
    beta = (xtx_inv @ xv.T @ yv).reshape(-1)
    resid = (yv.reshape(-1) - (xv @ beta))

    use_lag = nw_default_lag(n) if lag is None else int(max(0, lag))

    # Newey-West HAC covariance of score process.
    s = np.zeros((k, k), dtype=float)
    for t in range(n):
        xt = xv[t : t + 1, :].T
        s += (resid[t] ** 2) * (xt @ xt.T)
    for l in range(1, use_lag + 1):
        w = 1.0 - (l / (use_lag + 1.0))
        gamma = np.zeros((k, k), dtype=float)
        for t in range(l, n):
            xt = xv[t : t + 1, :].T
            xlag = xv[t - l : t - l + 1, :].T
            gamma += resid[t] * resid[t - l] * (xt @ xlag.T)
        s += w * (gamma + gamma.T)

    v_hac = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(v_hac), 0.0))
    tvals = beta / se
    pvals = np.array([math.erfc(abs(float(tv)) / math.sqrt(2.0)) if np.isfinite(tv) else np.nan for tv in tvals])

    yhat = xv @ beta
    sse = float(np.sum((yv.reshape(-1) - yhat) ** 2))
    sst = float(np.sum((yv.reshape(-1) - float(np.mean(yv))) ** 2))
    r2 = (1.0 - sse / sst) if sst > 0 else np.nan

    return {
        "beta": beta,
        "se_hac": se,
        "t_hac": tvals,
        "p_two_sided_norm": pvals,
        "r2": r2,
        "lag": float(use_lag),
    }


def _extend_unique(base: list[str], add: list[str]) -> list[str]:
    seen = set(base)
    out = list(base)
    for sym in add:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def build_active_baskets(
    include_core: bool = True,
    add_sector: bool = False,
    add_factor: bool = False,
    add_fixed_income: bool = False,
) -> list[QuadBasket]:
    out: list[QuadBasket] = []
    for b in CORE_QUAD_BASKETS:
        longs: list[str] = []
        shorts: list[str] = []

        if include_core:
            ex = CATEGORY_BUCKETS["core"].get(b.name, {})
            longs = _extend_unique(longs, ex.get("longs", []))
            shorts = _extend_unique(shorts, ex.get("shorts", []))
        if add_sector:
            ex = CATEGORY_BUCKETS["sector"].get(b.name, {})
            longs = _extend_unique(longs, ex.get("longs", []))
            shorts = _extend_unique(shorts, ex.get("shorts", []))
        if add_factor:
            ex = CATEGORY_BUCKETS["factor"].get(b.name, {})
            longs = _extend_unique(longs, ex.get("longs", []))
            shorts = _extend_unique(shorts, ex.get("shorts", []))
        if add_fixed_income:
            ex = CATEGORY_BUCKETS["fixed_income"].get(b.name, {})
            longs = _extend_unique(longs, ex.get("longs", []))
            shorts = _extend_unique(shorts, ex.get("shorts", []))

        out.append(QuadBasket(name=b.name, longs=longs, shorts=shorts))
    return out


def all_symbols(baskets: list[QuadBasket]) -> list[str]:
    seen: set[str] = set()
    syms: list[str] = []
    for b in baskets:
        for sym in b.longs + b.shorts:
            if sym not in seen:
                seen.add(sym)
                syms.append(sym)
    return syms


@st.cache_data(ttl=120)
def fetch_ohlc(symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        symbols,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, symbols[:1]])
    raw = raw.sort_index().dropna(how="all")
    return raw


@st.cache_data(ttl=120)
def fetch_benchmark_close(symbol: str, period: str, interval: str) -> pd.Series:
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        return pd.Series(dtype=float)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.Series(dtype=float)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            return pd.Series(dtype=float)
        close = raw["Close"]
    return close.sort_index().ffill().dropna()


def regular_hours_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize("America/New_York")
        else:
            out.index = out.index.tz_convert("America/New_York")
    except Exception:
        pass
    out = out[out.index.dayofweek < 5]
    try:
        out = out.between_time("09:30", "16:00")
    except Exception:
        pass
    return out


def _field_frame(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in raw.columns.get_level_values(0):
        return pd.DataFrame(index=raw.index)
    out = raw[field].copy()
    if isinstance(out, pd.Series):
        out = out.to_frame()
    return out


def build_quad_ohlc(raw: pd.DataFrame, baskets: list[QuadBasket], base: float = 100.0) -> dict[str, pd.DataFrame]:
    o = _field_frame(raw, "Open")
    h = _field_frame(raw, "High")
    l = _field_frame(raw, "Low")
    c = _field_frame(raw, "Close")
    if c.empty:
        return {}

    prev_c = c.shift(1)
    oret = (o / prev_c) - 1.0
    hret = (h / prev_c) - 1.0
    lret = (l / prev_c) - 1.0
    cret = (c / prev_c) - 1.0

    quad_ohlc: dict[str, pd.DataFrame] = {}

    for basket in baskets:
        longs = [s for s in basket.longs if s in c.columns]
        shorts = [s for s in basket.shorts if s in c.columns]
        if not longs or not shorts:
            continue

        r_long_open = oret[longs].mean(axis=1, skipna=True).fillna(0.0)
        r_long_high = hret[longs].mean(axis=1, skipna=True).fillna(0.0)
        r_long_low = lret[longs].mean(axis=1, skipna=True).fillna(0.0)
        r_long_close = cret[longs].mean(axis=1, skipna=True).fillna(0.0)
        r_short_open = oret[shorts].mean(axis=1, skipna=True).fillna(0.0)
        r_short_high = hret[shorts].mean(axis=1, skipna=True).fillna(0.0)
        r_short_low = lret[shorts].mean(axis=1, skipna=True).fillna(0.0)
        r_short_close = cret[shorts].mean(axis=1, skipna=True).fillna(0.0)

        idx = pd.DataFrame(index=c.index)
        prev_long = base
        prev_short = base
        long_open_vals: list[float] = []
        long_high_vals: list[float] = []
        long_low_vals: list[float] = []
        long_close_vals: list[float] = []
        short_open_vals: list[float] = []
        short_high_vals: list[float] = []
        short_low_vals: list[float] = []
        short_close_vals: list[float] = []
        open_vals: list[float] = []
        high_vals: list[float] = []
        low_vals: list[float] = []
        close_vals: list[float] = []

        for ts in idx.index:
            rl_o = float(r_long_open.loc[ts])
            rl_h = float(r_long_high.loc[ts])
            rl_l = float(r_long_low.loc[ts])
            rl_c = float(r_long_close.loc[ts])
            rs_o = float(r_short_open.loc[ts])
            rs_h = float(r_short_high.loc[ts])
            rs_l = float(r_short_low.loc[ts])
            rs_c = float(r_short_close.loc[ts])

            long_o = prev_long * (1.0 + rl_o)
            long_c = prev_long * (1.0 + rl_c)
            long_h = prev_long * (1.0 + (max(rl_h, rl_o, rl_c)))
            long_l = prev_long * (1.0 + (min(rl_l, rl_o, rl_c)))

            short_o = prev_short * (1.0 + rs_o)
            short_c = prev_short * (1.0 + rs_c)
            short_h = prev_short * (1.0 + (max(rs_h, rs_o, rs_c)))
            short_l = prev_short * (1.0 + (min(rs_l, rs_o, rs_c)))

            long_open_vals.append(float(long_o))
            long_high_vals.append(float(long_h))
            long_low_vals.append(float(long_l))
            long_close_vals.append(float(long_c))
            short_open_vals.append(float(short_o))
            short_high_vals.append(float(short_h))
            short_low_vals.append(float(short_l))
            short_close_vals.append(float(short_c))

            # total = 100 + (long_idx - 100) - (short_idx - 100)
            o_px = 100.0 + (long_o - 100.0) - (short_o - 100.0)
            c_px = 100.0 + (long_c - 100.0) - (short_c - 100.0)
            h_px = 100.0 + (long_h - 100.0) - (short_l - 100.0)
            l_px = 100.0 + (long_l - 100.0) - (short_h - 100.0)
            open_vals.append(float(o_px))
            high_vals.append(float(max(h_px, o_px, c_px)))
            low_vals.append(float(min(l_px, o_px, c_px)))
            close_vals.append(float(c_px))
            prev_long = long_c
            prev_short = short_c

        idx["LongOpen"] = long_open_vals
        idx["LongHigh"] = long_high_vals
        idx["LongLow"] = long_low_vals
        idx["LongClose"] = long_close_vals
        idx["ShortOpen"] = short_open_vals
        idx["ShortHigh"] = short_high_vals
        idx["ShortLow"] = short_low_vals
        idx["ShortClose"] = short_close_vals
        idx["Open"] = open_vals
        idx["High"] = high_vals
        idx["Low"] = low_vals
        idx["Close"] = close_vals
        idx = idx.dropna(how="any")

        if not idx.empty:
            # Rebase both legs to start at 100, then rebuild total from the same leg paths.
            long_scale = 100.0 / float(idx["LongClose"].iloc[0])
            short_scale = 100.0 / float(idx["ShortClose"].iloc[0])
            for ccol in ["LongOpen", "LongHigh", "LongLow", "LongClose"]:
                idx[ccol] = idx[ccol] * long_scale
            for ccol in ["ShortOpen", "ShortHigh", "ShortLow", "ShortClose"]:
                idx[ccol] = idx[ccol] * short_scale

            idx["Open"] = 100.0 + (idx["LongOpen"] - 100.0) - (idx["ShortOpen"] - 100.0)
            idx["Close"] = 100.0 + (idx["LongClose"] - 100.0) - (idx["ShortClose"] - 100.0)
            idx["High"] = 100.0 + (idx["LongHigh"] - 100.0) - (idx["ShortLow"] - 100.0)
            idx["Low"] = 100.0 + (idx["LongLow"] - 100.0) - (idx["ShortHigh"] - 100.0)
            idx["High"] = idx[["High", "Open", "Close"]].max(axis=1)
            idx["Low"] = idx[["Low", "Open", "Close"]].min(axis=1)

        quad_ohlc[basket.name] = idx

    return quad_ohlc


def plot_quad_candles(quad_ohlc: dict[str, pd.DataFrame], title: str, compress_time: bool = False) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[quad_label(x) for x in quad_ohlc.keys()],
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, (name, df) in enumerate(quad_ohlc.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        color = quad_color(name)

        x_vals = df.index.strftime("%m-%d %H:%M") if compress_time else df.index
        fig.add_trace(
            go.Candlestick(
                x=x_vals,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=quad_label(name),
                increasing_line_color=color,
                decreasing_line_color="#7f8c8d",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        if compress_time:
            tick_vals = sampled_tickvals(x_vals, max_labels=10)
            fig.update_xaxes(
                type="category",
                tickangle=-30,
                tickmode="array",
                tickvals=tick_vals,
                tickfont=dict(size=10),
                row=row,
                col=col,
            )
        else:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                ],
                row=row,
                col=col,
            )

    fig.update_layout(
        title=title,
        height=900,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        xaxis4_rangeslider_visible=False,
    )
    return fig


def plot_line_overlay(quad_ohlc: dict[str, pd.DataFrame], title: str, compress_time: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, df in quad_ohlc.items():
        x_vals = df.index.strftime("%m-%d %H:%M") if compress_time else df.index
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df["Close"],
                mode="lines",
                name=quad_label(name),
                line=dict(color=quad_color(name), width=2),
                connectgaps=False,
            )
        )

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(title=title, height=520, template="plotly_white", yaxis_title="Normalized Index")
    if compress_time:
        sample_x = None
        for _, df in quad_ohlc.items():
            if not df.empty:
                sample_x = df.index.strftime("%m-%d %H:%M")
                break
        tick_vals = sampled_tickvals(sample_x, max_labels=14) if sample_x is not None else None
        fig.update_xaxes(
            type="category",
            tickangle=-30,
            tickmode="array",
            tickvals=tick_vals,
            tickfont=dict(size=10),
        )
    else:
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ]
        )
    return fig


def build_snapshot_table(quad_ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in quad_ohlc.items():
        if df.empty:
            continue
        close = df["Close"].dropna()
        if close.empty:
            continue
        last = float(close.iloc[-1])
        # Session-level closes (one per trading date).
        sess_close = close.groupby(close.index.date).last()
        sess_vals = sess_close.values

        def pct_from_n_sessions_back(n: int) -> float:
            if len(sess_vals) <= n:
                return float("nan")
            prev_val = float(sess_vals[-(n + 1)])
            return (last / prev_val - 1.0) * 100.0

        dod = pct_from_n_sessions_back(1)
        wow = pct_from_n_sessions_back(5)
        mom = pct_from_n_sessions_back(21)
        total = (last / float(close.iloc[0]) - 1.0) * 100.0
        rows.append(
            {
                "Quad": quad_label(name),
                "Last": last,
                "DoD%": dod,
                "WoW%": wow,
                "MoM%": mom,
                "Total Period": total,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Total Period", ascending=False).reset_index(drop=True)


def style_signed_cols(v: float) -> str:
    if pd.isna(v):
        return ""
    if v > 0:
        return "color: #1db954; font-weight: 600;"
    if v < 0:
        return "color: #ff4d4f; font-weight: 600;"
    return "color: #b0b0b0;"


def quad_session_return_map(quad_ohlc: dict[str, pd.DataFrame]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, df in quad_ohlc.items():
        if df.empty:
            continue
        first = float(df["Close"].iloc[0])
        last = float(df["Close"].iloc[-1])
        out[name] = ((last / first) - 1.0) * 100.0
    return out


def quad_leg_performance_maps_from_quad_ohlc(
    quad_ohlc: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    long_map: dict[str, float] = {}
    short_map: dict[str, float] = {}
    total_map: dict[str, float] = {}
    for name, df in quad_ohlc.items():
        if df.empty or "LongClose" not in df.columns or "ShortClose" not in df.columns:
            continue
        long_pct = (float(df["LongClose"].iloc[-1]) / float(df["LongClose"].iloc[0]) - 1.0) * 100.0
        short_pct = (float(df["ShortClose"].iloc[-1]) / float(df["ShortClose"].iloc[0]) - 1.0) * 100.0
        total_pct = long_pct - short_pct
        long_map[name] = long_pct
        short_map[name] = short_pct
        total_map[name] = total_pct
    return long_map, short_map, total_map


def build_component_rebased_close(raw: pd.DataFrame, basket: QuadBasket, base: float = 100.0) -> pd.DataFrame:
    close = _field_frame(raw, "Close")
    cols = [s for s in basket.longs + basket.shorts if s in close.columns]
    if not cols:
        return pd.DataFrame()
    out = close[cols].copy().dropna(how="all")
    out = out.ffill()
    out = out.div(out.iloc[0]).mul(base)
    return out.dropna(how="all")


def signflip_pvalue(values: np.ndarray, n_perm: int = 2000, seed: int = 42, batch: int = 250) -> float:
    if values.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    extreme = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, values.size))
        perm_means = np.mean(signs * values, axis=1)
        extreme += int(np.sum(np.abs(perm_means) >= abs(observed)))
        done += b
    return (extreme + 1.0) / (n_perm + 1.0)


def signflip_pvalue_right_tail(values: np.ndarray, n_perm: int = 2000, seed: int = 42, batch: int = 250) -> float:
    if values.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = float(np.mean(values))
    extreme = 0
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, values.size))
        perm_means = np.mean(signs * values, axis=1)
        extreme += int(np.sum(perm_means >= observed))
        done += b
    return (extreme + 1.0) / (n_perm + 1.0)


def build_quad_alpha_stats(
    raw: pd.DataFrame,
    baskets: list[QuadBasket],
    benchmark_close: pd.Series | None = None,
    n_perm: int = 2000,
    ppy: float = 252.0,
    sig_basis: str = "Weekly",
) -> pd.DataFrame:
    close = _field_frame(raw, "Close").ffill().dropna(how="all")
    if close.empty:
        return pd.DataFrame()

    ret = close.pct_change().dropna(how="all")
    bench_ret_full = None
    if benchmark_close is not None and not benchmark_close.empty:
        bench_ret_full = benchmark_close.sort_index().ffill().pct_change()

    rows: list[dict[str, float | str | bool]] = []
    for basket in baskets:
        longs = [s for s in basket.longs if s in ret.columns]
        shorts = [s for s in basket.shorts if s in ret.columns]
        if not longs or not shorts:
            continue

        long_ret = ret[longs].mean(axis=1, skipna=True).fillna(0.0)
        short_ret = ret[shorts].mean(axis=1, skipna=True).fillna(0.0)
        spread = (long_ret - short_ret).fillna(0.0)
        if spread.empty:
            continue

        common = spread.index
        long_c = long_ret.loc[common]
        short_c = short_ret.loc[common]
        bench_ret_common = None
        bench_pct = float("nan")
        if bench_ret_full is not None:
            bench_ret_common = bench_ret_full.reindex(common).fillna(0.0)
            bench_pct = ((1.0 + bench_ret_common).prod() - 1.0) * 100.0

        # Significance is computed on selected paired spread basis.
        long_px = close[longs].mean(axis=1, skipna=True).ffill()
        short_px = close[shorts].mean(axis=1, skipna=True).ffill()
        basis_to_rule = {"Hourly": "1H", "Daily": "1D", "Weekly": "W-FRI"}
        rule = basis_to_rule.get(sig_basis, "W-FRI")
        basis_long = long_px.resample(rule).last().pct_change().dropna()
        basis_short = short_px.resample(rule).last().pct_change().dropna()
        basis_spread = (basis_long - basis_short).dropna()
        if basis_spread.size >= 3:
            spread_sig = basis_spread
        else:
            # Fallback when the selected window is too short for basis stats.
            spread_sig = spread

        n = int(spread_sig.size)
        mean = float(spread_sig.mean())
        std = float(spread_sig.std(ddof=1)) if n > 1 else float("nan")
        stderr = (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
        t_stat = (mean / stderr) if n > 1 and stderr > 0 else float("nan")
        p_pair_2s = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else float("nan")
        sig_pair_90 = bool(p_pair_2s < 0.10) if np.isfinite(p_pair_2s) else False
        long_pct = ((1.0 + long_c).prod() - 1.0) * 100.0
        short_pct = ((1.0 + short_c).prod() - 1.0) * 100.0
        spread_pct = long_pct - short_pct
        basket_sharpe = annualized_sharpe(spread, ppy=ppy)
        benchmark_sharpe = annualized_sharpe(bench_ret_common, ppy=ppy) if bench_ret_common is not None else float("nan")
        corr_long_short = long_c.corr(short_c)

        rows.append(
            {
                "Quad": basket.name,
                "Bars": n,
                "LongRetPct": long_pct,
                "ShortRetPct": short_pct,
                "BenchmarkRetPct": bench_pct,
                "AlphaSpreadPct": spread_pct,
                "BasketSharpe": basket_sharpe,
                "BenchmarkSharpe": benchmark_sharpe,
                "CorrLongShort": corr_long_short,
                "PairDiffTStat": t_stat,
                "PairDiffPValue2S": p_pair_2s,
                "PairDiffSig90": sig_pair_90,
                "SigBasis": sig_basis if basis_spread.size >= 3 else "IntradayFallback",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("AlphaSpreadPct", ascending=False).reset_index(drop=True)


def build_cross_quad_significance(
    quad_ohlc: dict[str, pd.DataFrame],
    sig_basis: str = "Weekly",
) -> pd.DataFrame:
    if not quad_ohlc:
        return pd.DataFrame()

    close_map = {}
    for name, df in quad_ohlc.items():
        if df.empty or "Close" not in df.columns:
            continue
        close_map[quad_label(name)] = df["Close"].copy()
    if len(close_map) < 2:
        return pd.DataFrame()

    close_df = pd.DataFrame(close_map).dropna(how="all").ffill().dropna(how="all")
    if close_df.empty:
        return pd.DataFrame()

    basis_to_rule = {"Hourly": "1H", "Daily": "1D", "Weekly": "W-FRI"}
    rule = basis_to_rule.get(sig_basis, "W-FRI")
    basis_ret = close_df.resample(rule).last().pct_change().dropna(how="all")
    if basis_ret.empty:
        basis_ret = close_df.pct_change().dropna(how="all")
    basis_ret = basis_ret.fillna(0.0)

    totals = ((close_df.iloc[-1] / close_df.iloc[0]) - 1.0) * 100.0
    rows: list[dict[str, float | str | bool]] = []
    cols = list(basis_ret.columns)
    for q in cols:
        others = [c for c in cols if c != q]
        if not others:
            continue
        diff = (basis_ret[q] - basis_ret[others].mean(axis=1, skipna=True)).dropna()
        n = int(diff.size)
        mean = float(diff.mean()) if n > 0 else float("nan")
        std = float(diff.std(ddof=1)) if n > 1 else float("nan")
        stderr = (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
        t_stat = (mean / stderr) if n > 1 and stderr > 0 else float("nan")
        p_two = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else float("nan")

        rest_total = float(totals[others].mean()) if others else float("nan")
        rows.append(
            {
                "Quad": q,
                "Total Perf %": float(totals[q]),
                "Rest Avg Perf %": rest_total,
                "Gap vs Rest %": float(totals[q] - rest_total),
                "Cross-Quad p-value": p_two,
                "Cross Sig90": bool(p_two < 0.10) if np.isfinite(p_two) else False,
                "BasisObs": n,
                "SigBasis": sig_basis,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Gap vs Rest %", ascending=False).reset_index(drop=True)


def build_quad_correlation_matrix(
    quad_ohlc: dict[str, pd.DataFrame],
    basis: str = "Weekly",
) -> pd.DataFrame:
    if not quad_ohlc:
        return pd.DataFrame()
    close_map = {}
    for name, df in quad_ohlc.items():
        if df.empty or "Close" not in df.columns:
            continue
        close_map[quad_label(name)] = df["Close"].copy()
    if len(close_map) < 2:
        return pd.DataFrame()

    close_df = pd.DataFrame(close_map).dropna(how="all").ffill().dropna(how="all")
    if close_df.empty:
        return pd.DataFrame()

    basis_to_rule = {"Hourly": "1H", "Daily": "1D", "Weekly": "W-FRI"}
    rule = basis_to_rule.get(basis, "W-FRI")
    ret = close_df.resample(rule).last().pct_change().dropna(how="all")
    if ret.empty:
        ret = close_df.pct_change().dropna(how="all")
    ret = ret.fillna(0.0)
    if ret.empty:
        return pd.DataFrame()
    return ret.corr()


def build_quad_vs_benchmark_stats(
    quad_ohlc: dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    basis: str = "Weekly",
) -> pd.DataFrame:
    if not quad_ohlc or benchmark_close is None or benchmark_close.empty:
        return pd.DataFrame()

    basis_to_rule = {"Hourly": "1H", "Daily": "1D", "Weekly": "W-FRI"}
    rule = basis_to_rule.get(basis, "W-FRI")
    bench = benchmark_close.dropna().copy()
    bench_ret = bench.resample(rule).last().pct_change().dropna()
    if bench_ret.empty:
        bench_ret = bench.pct_change().dropna()
    if bench_ret.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | bool]] = []
    for qname, qdf in quad_ohlc.items():
        if qdf.empty or "Close" not in qdf.columns:
            continue
        q = qdf["Close"].dropna()
        q_ret = q.resample(rule).last().pct_change().dropna()
        if q_ret.empty:
            q_ret = q.pct_change().dropna()
        aligned = pd.concat([q_ret.rename("q"), bench_ret.rename("b")], axis=1).dropna()
        if aligned.empty:
            continue

        corr = float(aligned["q"].corr(aligned["b"]))
        b_var = float(aligned["b"].var(ddof=1))
        beta = float(aligned["q"].cov(aligned["b"]) / b_var) if b_var > 0 else float("nan")

        diff = aligned["q"] - aligned["b"]
        n = int(diff.size)
        mean = float(diff.mean()) if n > 0 else float("nan")
        std = float(diff.std(ddof=1)) if n > 1 else float("nan")
        stderr = (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
        t_stat = (mean / stderr) if n > 1 and stderr > 0 else float("nan")
        p_two = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else float("nan")

        rows.append(
            {
                "Quad": quad_label(qname),
                "CorrVsBenchmark": corr,
                "BetaVsBenchmark": beta,
                "QuadVsBenchmarkPValue2S": p_two,
                "QuadVsBenchmarkSig90": bool(p_two < 0.10) if np.isfinite(p_two) else False,
                "BasisObsQB": n,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def plot_quad_corr_heatmap(corr: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if corr.empty:
        fig.update_layout(title=f"{title} (No Data)", template="plotly_white", height=420)
        return fig

    z = corr.values.astype(float)
    labels = corr.columns.tolist()
    text = [[f"{v:+.2f}" for v in row] for row in z]
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale=[
                [0.0, "#d73027"],   # red
                [0.5, "#ffffff"],   # white
                [1.0, "#1a9850"],   # green
            ],
            colorbar=dict(title="Corr"),
            text=text,
            texttemplate="%{text}",
            hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=520,
        xaxis=dict(tickangle=-20),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def all_universe_symbols() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bucket in CATEGORY_BUCKETS.values():
        for q in bucket.values():
            for sym in q.get("longs", []) + q.get("shorts", []):
                if sym not in seen:
                    seen.add(sym)
                    out.append(sym)
    return out


def build_quad_stance_map(baskets: list[QuadBasket]) -> dict[str, dict[str, int]]:
    stance: dict[str, dict[str, int]] = {}
    for b in baskets:
        qmap: dict[str, int] = {}
        for s in b.longs:
            qmap[s] = 1
        for s in b.shorts:
            qmap[s] = -1
        stance[b.name] = qmap
    return stance


def build_macro_heatmap_data(close: pd.DataFrame, baskets: list[QuadBasket]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if close.empty:
        return pd.DataFrame(), pd.DataFrame()
    first = close.ffill().bfill().iloc[0]
    last = close.ffill().iloc[-1]
    ret_pct = ((last / first) - 1.0) * 100.0
    raw_df = ret_pct.to_frame("ReturnPct").sort_index()

    stance = build_quad_stance_map(baskets)
    score = pd.DataFrame(index=raw_df.index)
    for qname, smap in stance.items():
        vals = []
        for sym in raw_df.index:
            stn = smap.get(sym, 0)
            vals.append(float(raw_df.loc[sym, "ReturnPct"]) * stn if stn != 0 else np.nan)
        score[qname] = vals
    return raw_df, score


def plot_macro_score_heatmap(score: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if score.empty:
        fig.update_layout(title=f"{title} (No Data)", template="plotly_white", height=420)
        return fig
    z = score.values
    x = score.columns.tolist()
    y = score.index.tolist()
    text = [[("" if np.isnan(v) else f"{v:+.1f}") for v in row] for row in z]
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0.0, "#d73027"], [0.5, "#ffffff"], [1.0, "#1a9850"]],
            zmid=0.0,
            text=text,
            texttemplate="%{text}",
            hovertemplate="ETF=%{y}<br>Quad=%{x}<br>Score=%{z:.2f}%<extra></extra>",
            colorbar=dict(title="Signed %"),
        )
    )
    fig.update_layout(title=title, template="plotly_white", height=900, yaxis=dict(autorange="reversed"))
    return fig


def plot_quad_alpha_bars(df: pd.DataFrame, y_col: str, x_col: str = "Quad") -> go.Figure:
    fig = go.Figure()
    if df.empty or y_col not in df.columns or x_col not in df.columns:
        fig.update_layout(title="Quad Alpha Spread (No Data)", template="plotly_white", height=380)
        return fig
    alpha_vals = df[y_col].astype(float)
    bar_text = [f"{v:+.4f}%" if abs(v) < 0.1 else f"{v:+.2f}%" for v in alpha_vals]
    fig.add_trace(
        go.Bar(
            x=df[x_col].map(quad_label) if hasattr(df[x_col], "map") else df[x_col],
            y=alpha_vals,
            marker_color=[quad_color(x) for x in df[x_col]],
            text=bar_text,
            textposition="outside",
            name="Alpha Spread",
        )
    )
    fig.update_layout(
        title="Absolute Quad Alpha Spread (Long Basket minus Short Basket)",
        template="plotly_white",
        height=420,
        yaxis_title="Cumulative Spread Return (%)",
    )
    y_min = float(alpha_vals.min())
    y_max = float(alpha_vals.max())
    pad = max(1.0, (y_max - y_min) * 0.15)
    fig.update_yaxes(range=[y_min - pad, y_max + pad])
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    return fig


def plot_components_overlay(
    raw: pd.DataFrame,
    basket: QuadBasket,
    title: str,
    basket_total: pd.DataFrame | None = None,
    compress_time: bool = False,
) -> go.Figure:
    df = build_component_rebased_close(raw, basket, base=100.0)
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{title} (No component data)", template="plotly_white", height=420)
        return fig

    x_vals = df.index.strftime("%m-%d %H:%M") if compress_time else df.index
    for sym in basket.longs:
        if sym in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=df[sym],
                    mode="lines",
                    name=f"{sym} (L)",
                    line=dict(color="#2ca25f", width=2),
                    opacity=0.9,
                )
            )
    for sym in basket.shorts:
        if sym in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=df[sym],
                    mode="lines",
                    name=f"{sym} (S)",
                    line=dict(color="#d62728", width=2, dash="dot"),
                    opacity=0.9,
                )
            )

    if basket_total is not None and not basket_total.empty and "Close" in basket_total.columns:
        x_total = basket_total.index.strftime("%m-%d %H:%M") if compress_time else basket_total.index
        fig.add_trace(
            go.Scatter(
                x=x_total,
                y=basket_total["Close"],
                mode="lines",
                name="Basket Total",
                line=dict(color="#00d4ff", width=3),
                opacity=1.0,
            )
        )

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=title,
        height=460,
        template="plotly_white",
        yaxis_title="Rebased Price (Start = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if compress_time:
        tick_vals = sampled_tickvals(x_vals, max_labels=14)
        fig.update_xaxes(
            type="category",
            tickangle=-30,
            tickmode="array",
            tickvals=tick_vals,
            tickfont=dict(size=10),
        )
    else:
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ]
        )
    return fig


def main() -> None:
    st.set_page_config(page_title="Quad Baskets Candlestick Monitor", layout="wide")
    st.title("Quad Basket Candlestick Monitor")
    st.caption("Synthetic quad OHLC built from hardwired long-short ETF basket returns.")

    with st.sidebar:
        st.header("Controls")
        period = st.selectbox("History", ["1d", "5d", "10d", "1mo", "3mo", "6mo", "1y"], index=3)
        if period in {"1d", "5d", "10d", "1mo"}:
            interval = st.selectbox("Bar Interval", ["5m", "30m"], index=0)
        else:
            interval = st.selectbox("Bar Interval", ["60m"], index=0, disabled=True)
        st.markdown("`Universe`")
        include_core = st.checkbox("Include CORE", value=True)
        add_sector = st.checkbox("Add All Sectors", value=False)
        add_factor = st.checkbox("Add All Factors", value=False)
        add_fixed_income = st.checkbox("Add All Fixed Income", value=False)
        benchmark_symbol = st.selectbox("Benchmark", ["SPY", "^GSPC"], index=0)
        sig_basis = st.selectbox("Sig Basis", ["Weekly", "Daily", "Hourly"], index=0)
        rth_only = st.checkbox("Regular Trading Hours only", value=True)
        compress_time = st.checkbox("Compress time (remove no-data gaps)", value=True)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_seconds = st.number_input("Refresh seconds", min_value=5, max_value=300, value=30, step=5)
        do_refresh = st.button("Refresh")

    if do_refresh:
        st.cache_data.clear()

    if auto_refresh:
        if st_autorefresh is not None:
            st_autorefresh(interval=int(refresh_seconds) * 1000, key="quad_auto_refresh")
            st.caption(f"Auto-refresh active: every {int(refresh_seconds)}s")
        else:
            st.warning(
                "Auto-refresh requires package `streamlit-autorefresh`. "
                "Install with: pip install streamlit-autorefresh"
            )

    active_baskets = build_active_baskets(
        include_core=include_core,
        add_sector=add_sector,
        add_factor=add_factor,
        add_fixed_income=add_fixed_income,
    )
    valid_basket_count = sum(1 for b in active_baskets if len(b.longs) > 0 and len(b.shorts) > 0)
    if valid_basket_count == 0:
        st.error("No valid baskets (need both longs and shorts). Enable CORE or add more universe groups.")
        return
    symbols = all_symbols(active_baskets)
    raw = fetch_ohlc(symbols, period=period, interval=interval)
    if raw.empty:
        st.error("No data returned from Yahoo Finance for this selection.")
        return

    if rth_only:
        raw = regular_hours_only(raw)

    benchmark_close = fetch_benchmark_close(benchmark_symbol, period=period, interval=interval)
    if rth_only and not benchmark_close.empty:
        benchmark_close = regular_hours_only(benchmark_close.to_frame("Close"))["Close"]
    if not benchmark_close.empty:
        benchmark_close = benchmark_close.reindex(raw.index).ffill().dropna()

    # Data freshness / timestamp check for intraday reconciliation.
    if len(raw.index) > 0:
        last_ts = raw.index.max()
        st.caption(f"Latest bar timestamp (ET): {last_ts}")

    ppy = periods_per_year(interval)

    quad_ohlc = build_quad_ohlc(raw, baskets=active_baskets, base=100.0)
    if not quad_ohlc:
        st.error("Could not build quad OHLC. Check symbol coverage for selected timeframe.")
        return

    st.plotly_chart(
        plot_quad_candles(
            quad_ohlc,
            title=f"Quad Basket Candlesticks ({interval}, {period})",
            compress_time=compress_time,
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            plot_line_overlay(
                quad_ohlc,
                title="Normalized Close Overlay (Start = 100)",
                compress_time=compress_time,
            ),
            use_container_width=True,
        )
    with col2:
        snap = build_snapshot_table(quad_ohlc)
        st.subheader("Latest Snapshot")
        if snap.empty:
            st.write("No snapshot rows.")
        else:
            st.dataframe(
                snap.style.format(
                    {
                        "Last": "{:.2f}",
                        "DoD%": "{:+.2f}%",
                        "WoW%": "{:+.2f}%",
                        "MoM%": "{:+.2f}%",
                        "Total Period": "{:+.2f}%",
                    }
                ).applymap(style_signed_cols, subset=["DoD%", "WoW%", "MoM%", "Total Period"]),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Quad Alpha / Statistical Significance")
    alpha_stats = build_quad_alpha_stats(
        raw,
        baskets=active_baskets,
        benchmark_close=benchmark_close,
        n_perm=2000,
        ppy=ppy,
        sig_basis=sig_basis,
    )
    if alpha_stats.empty:
        st.write("No alpha stats available for current filter.")
    else:
        long_map, short_map, total_map = quad_leg_performance_maps_from_quad_ohlc(quad_ohlc)
        perf = alpha_stats.copy()
        if "PairDiffSig90" not in perf.columns and "PairDiffSig95" in perf.columns:
            perf["PairDiffSig90"] = perf["PairDiffSig95"]
        perf = perf[
            [
                "Quad",
                "LongRetPct",
                "ShortRetPct",
                "AlphaSpreadPct",
                "BenchmarkRetPct",
                "BasketSharpe",
                "BenchmarkSharpe",
                "PairDiffPValue2S",
                "PairDiffSig90",
            ]
        ].rename(
            columns={
                "LongRetPct": "Long Perf %",
                "ShortRetPct": "Short Perf %",
                "AlphaSpreadPct": "Total Basket Performance %",
                "BenchmarkRetPct": f"{benchmark_symbol} Performance %",
                "BasketSharpe": "Sharpe",
                "BenchmarkSharpe": f"{benchmark_symbol} Sharpe",
                "PairDiffPValue2S": "Significance p-value",
                "PairDiffSig90": "Sig90",
            }
        )
        # Canonical reconciliation: all three metrics from the same leg-performance maps.
        perf["Long Perf %"] = perf["Quad"].map(long_map).fillna(perf["Long Perf %"])
        perf["Short Perf %"] = perf["Quad"].map(short_map).fillna(perf["Short Perf %"])
        perf["Total Basket Performance %"] = perf["Quad"].map(total_map).fillna(
            perf["Long Perf %"] - perf["Short Perf %"]
        )
        perf["Quad"] = perf["Quad"].map(quad_label)
        qb = build_quad_vs_benchmark_stats(quad_ohlc=quad_ohlc, benchmark_close=benchmark_close, basis=sig_basis)
        if not qb.empty:
            perf = perf.merge(
                qb[
                    [
                        "Quad",
                        "CorrVsBenchmark",
                        "BetaVsBenchmark",
                        "QuadVsBenchmarkPValue2S",
                        "QuadVsBenchmarkSig90",
                    ]
                ],
                on="Quad",
                how="left",
            )
            perf = perf.rename(
                columns={
                    "CorrVsBenchmark": f"Corr vs {benchmark_symbol}",
                    "BetaVsBenchmark": f"Beta vs {benchmark_symbol}",
                    "QuadVsBenchmarkPValue2S": f"Diff vs {benchmark_symbol} p-value",
                    "QuadVsBenchmarkSig90": f"Diff vs {benchmark_symbol} Sig90",
                }
            )
        perf = perf.sort_values("Total Basket Performance %", ascending=False).reset_index(drop=True)
        st.plotly_chart(
            plot_quad_alpha_bars(perf, y_col="Total Basket Performance %", x_col="Quad"),
            use_container_width=True,
        )

        st.dataframe(
            perf.style.format(
                {
                    "Long Perf %": "{:+.2f}%",
                    "Short Perf %": "{:+.2f}%",
                    "Total Basket Performance %": "{:+.4f}%",
                    f"{benchmark_symbol} Performance %": "{:+.2f}%",
                    "Sharpe": "{:+.2f}",
                    f"{benchmark_symbol} Sharpe": "{:+.2f}",
                    "Significance p-value": "{:.4f}",
                    f"Corr vs {benchmark_symbol}": "{:+.2f}",
                    f"Beta vs {benchmark_symbol}": "{:+.2f}",
                    f"Diff vs {benchmark_symbol} p-value": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Benchmark column is based on selected symbol: {benchmark_symbol}. "
            f"Significance p-value / Sig90 use paired tests on {sig_basis.upper()} spread returns."
        )

        st.subheader("Cross-Quad Significance (Each Quad vs Other 3)")
        cross = build_cross_quad_significance(quad_ohlc=quad_ohlc, sig_basis=sig_basis)
        if cross.empty:
            st.write("No cross-quad significance rows.")
        else:
            st.dataframe(
                cross.style.format(
                    {
                        "Total Perf %": "{:+.4f}%",
                        "Rest Avg Perf %": "{:+.4f}%",
                        "Gap vs Rest %": "{:+.4f}%",
                        "Cross-Quad p-value": "{:.4f}",
                        "BasisObs": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Quad Correlation Heatmap")
        corr = build_quad_correlation_matrix(quad_ohlc=quad_ohlc, basis=sig_basis)
        st.plotly_chart(
            plot_quad_corr_heatmap(corr, title=f"Quad Correlation Matrix ({sig_basis} returns)"),
            use_container_width=True,
        )

    st.subheader("Per-Quad Components")
    tab_labels = [b.name for b in active_baskets]
    tab_labels = [quad_label(x) for x in tab_labels]
    quad_tabs = st.tabs(tab_labels)
    for tab, basket in zip(quad_tabs, active_baskets):
        with tab:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("`Longs`")
                st.write(", ".join(basket.longs))
            with c2:
                st.markdown("`Shorts`")
                st.write(", ".join(basket.shorts))

            st.plotly_chart(
                plot_components_overlay(
                    raw,
                    basket,
                    title=f"{quad_label(basket.name)} Components (Long=Green, Short=Red)",
                    basket_total=quad_ohlc.get(basket.name),
                    compress_time=compress_time,
                ),
                use_container_width=True,
            )

    st.markdown(
        "Run locally: "
        "`streamlit run '/Users/jacksolasz/Desktop/Projects/Desktop Inbox 2026-02-19/quad_streamlit_baskets.py'`"
    )


if __name__ == "__main__":
    main()
