#!/usr/bin/env python3
"""prepare.py - build the public/private files for the task
"Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records".

Pipeline (fully deterministic, base seed 20260707):

1.  Read the genuine raw daily OHLCV table (`prices.csv`, produced by
    `download_data.py` from a public CC BY 4.0 source).
2.  Select 160 assets with >= 600 contiguous, clean daily bars; carve one
    590-day window per asset ("arena"): steps 0-449 are the public
    in-sample segment, steps 450-589 the hidden continuation segment.
3.  Anonymize and re-scale each arena: prices divided by the first
    in-sample close, volume by the median in-sample volume, and every
    day's OHLC multiplied by a deterministic jitter factor in
    [0.998, 1.002] (volume jitter [0.98, 1.02]) so windows cannot be
    byte-matched against public archives. All simulation happens on the
    jittered series, so the task is internally consistent.
4.  Simulate rule-based long/flat trading strategies per arena:
      - 15 "informed" strategies with a persistent, regime-conditional,
        trade-level edge (apply_edge: losing base trades are vetoed and
        well-timed multi-day holds are inserted, both concentrated on
        high-volatility days) -> the edge survives into the hidden
        continuation segment because the same mechanics run there;
      - a pool of 150 pure-noise strategies from which 15 "lucky" ones
        are selected per arena as the nearest unused candidates in
        z-normalized (in-sample Sharpe, in-sample win rate) space, per
        family -> in-sample summary statistics are largely
        uninformative about the skill flag;
      - 8 "background" noise strategies (not matched).
5.  Emit public files (arena bars for steps 0-449, in-sample trade logs,
    labeled train strategies, unlabeled test strategies, sample
    submission) and the private answer key (skill flag, realized
    out-of-sample Sharpe on steps 450-589, bucket columns).

Arena split: 96 train / 64 test arenas; a strategy belongs to its
arena's split, so no test arena leaks its continuation regime through
train labels.

The platform calls:  prepare(dataset_dir, public_dir, private_dir)

Robust input handling: the raw table is searched in several standard
locations; if it is absent but prepared outputs already exist, they are
copied to the requested folders; if neither exists, the script attempts
the documented public download via download_data.py.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_SEED = 20260707

# ---- window geometry ----
IS_STEPS = 450          # public in-sample bars per arena (steps 0..449)
OOS_STEPS = 140         # hidden continuation bars (steps 450..589)
TOTAL_STEPS = IS_STEPS + OOS_STEPS
MIN_ROWS = 600          # eligibility threshold per asset

# ---- population design ----
N_ARENAS = 160
N_TRAIN_ARENAS = 96
N_INFORMED = 15         # per arena
N_LUCK = 15             # per arena (matched noise)
N_POOL = 150            # noise candidates per arena used for matching
N_BACKGROUND = 8        # per arena (plain noise, not matched)
FAMILIES = ("trend", "breakout", "meanrev", "voltimer")

# ---- edge mechanics (trade-level, shape-preserving) ----
EDGE_MIN, EDGE_MAX = 0.35, 0.75      # per-strategy edge intensity
VETO_OFFREGIME_FACTOR = 3.0          # vetoes are 3x rarer off-regime
INSERT_RATE = 0.18                   # insertion prob per eligible flat day
INSERT_HOLD_MIN, INSERT_HOLD_MAX = 3, 10
INSERT_MIN_GAIN = 0.03               # only insert holds that gained > 3%
ELIGIBLE_VOL_QUANTILE = 0.60         # edge concentrates on high-vol days
VOL_WINDOW = 5

# ---- misc ----
ANNUALIZE = np.sqrt(365.0)
SHARPE_CLIP = 10.0
K_SELECT = 240                        # portfolio size in the submission
PRICE_JITTER = 0.002
VOLUME_JITTER = 0.02

PUBLIC_FILES = (
    "train.csv", "test.csv", "sample_submission.csv",
    "arenas.csv", "trades.csv",
)
PRIVATE_FILES = ("answers.csv",)


def _seed(tag: str) -> int:
    """Stable 63-bit seed derived from the base seed and a string tag."""
    digest = hashlib.sha256(f"{BASE_SEED}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(_seed(tag))


# --------------------------------------------------------------------------
# arena construction
# --------------------------------------------------------------------------

def load_prices(prices_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        prices_csv,
        usecols=["symbol", "open_time", "open", "high", "low", "close",
                 "volume_busd"],
        parse_dates=["open_time"],
    )
    return df.sort_values(["symbol", "open_time"], kind="mergesort")


def eligible_assets(prices: pd.DataFrame) -> list[str]:
    keep = []
    for sym, g in prices.groupby("symbol", sort=True):
        if len(g) < MIN_ROWS:
            continue
        if (g["open_time"].diff().dt.days.iloc[1:] != 1).any():
            continue
        vals = g[["open", "high", "low", "close", "volume_busd"]].to_numpy()
        if (vals <= 0).any() or not np.isfinite(vals).all():
            continue
        if (g["high"] < g["low"]).any():
            continue
        keep.append(sym)
    return keep


def build_arenas(prices: pd.DataFrame) -> dict[str, dict]:
    """Return {arena_id: {ohlcv arrays (jittered+normalized), masks}}."""
    assets = eligible_assets(prices)
    if len(assets) < N_ARENAS:
        raise RuntimeError(
            f"Need {N_ARENAS} eligible assets, found {len(assets)}."
        )
    pick_rng = _rng("asset-pick")
    chosen = sorted(pick_rng.choice(assets, size=N_ARENAS, replace=False))
    order = pick_rng.permutation(N_ARENAS)  # anonymize: shuffle id order
    arena_ids = {chosen[k]: f"A{order[k] + 1:03d}" for k in range(N_ARENAS)}

    arenas: dict[str, dict] = {}
    for sym in chosen:
        g = prices[prices["symbol"] == sym].reset_index(drop=True)
        arng = _rng(f"arena:{sym}")
        start = int(arng.integers(0, len(g) - TOTAL_STEPS + 1))
        w = g.iloc[start:start + TOTAL_STEPS].reset_index(drop=True)

        o = w["open"].to_numpy(float).copy()
        h = w["high"].to_numpy(float).copy()
        lo = w["low"].to_numpy(float).copy()
        c = w["close"].to_numpy(float).copy()
        v = w["volume_busd"].to_numpy(float).copy()

        pj = 1.0 + arng.uniform(-PRICE_JITTER, PRICE_JITTER, TOTAL_STEPS)
        vj = 1.0 + arng.uniform(-VOLUME_JITTER, VOLUME_JITTER, TOTAL_STEPS)
        o, h, lo, c = o * pj, h * pj, lo * pj, c * pj
        v = v * vj

        scale = c[0]
        vscale = np.median(v[:IS_STEPS])
        o, h, lo, c = o / scale, h / scale, lo / scale, c / scale
        v = v / vscale

        r = np.zeros(TOTAL_STEPS)
        r[1:] = c[1:] / c[:-1] - 1.0
        logret = np.zeros(TOTAL_STEPS)
        logret[1:] = np.log(c[1:] / c[:-1])
        vol5 = np.full(TOTAL_STEPS, np.nan)
        for t in range(VOL_WINDOW, TOTAL_STEPS):
            vol5[t] = np.std(logret[t - VOL_WINDOW + 1:t + 1])
        v_thr = np.nanquantile(vol5[:IS_STEPS], ELIGIBLE_VOL_QUANTILE)
        eligible = np.zeros(TOTAL_STEPS, dtype=bool)
        eligible[VOL_WINDOW:] = vol5[VOL_WINDOW:] >= v_thr

        arenas[arena_ids[sym]] = {
            "open": o, "high": h, "low": lo, "close": c, "volume": v,
            "ret": r, "logret": logret, "eligible": eligible,
        }
    return arenas


# --------------------------------------------------------------------------
# base strategy rules (long/flat; position decided at the close of day t)
# --------------------------------------------------------------------------

def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    m = (cs[w:] - cs[:-w]) / w
    m2 = (cs2[w:] - cs2[:-w]) / w
    out[w - 1:] = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return out


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(w - 1, len(x)):
        out[t] = x[t - w + 1:t + 1].max()
    return out


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for t in range(w - 1, len(x)):
        out[t] = x[t - w + 1:t + 1].min()
    return out


def base_positions(arena: dict, family: str, params: dict) -> np.ndarray:
    c, h, lo = arena["close"], arena["high"], arena["low"]
    T = len(c)
    pos = np.zeros(T, dtype=np.int8)

    if family == "trend":
        f, s = params["fast"], params["slow"]
        mf, ms = _roll_mean(c, f), _roll_mean(c, s)
        pos = ((mf > ms) & ~np.isnan(ms)).astype(np.int8)

    elif family == "breakout":
        L, M = params["enter"], params["exit"]
        hh = _roll_max(h, L)
        ll = _roll_min(lo, M)
        state = 0
        for t in range(T):
            if state == 0:
                if t >= L and c[t] > hh[t - 1]:
                    state = 1
            else:
                if t >= M and c[t] < ll[t - 1]:
                    state = 0
            pos[t] = state

    elif family == "meanrev":
        w, z_in, z_out = params["window"], params["z_in"], params["z_out"]
        m, s = _roll_mean(c, w), _roll_std(c, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (c - m) / s
        state = 0
        for t in range(T):
            if np.isnan(z[t]):
                pos[t] = state
                continue
            if state == 0 and z[t] < -z_in:
                state = 1
            elif state == 1 and z[t] > -z_out:
                state = 0
            pos[t] = state

    elif family == "voltimer":
        vw, q, mw = params["vol_window"], params["quantile"], params["mom"]
        vol = _roll_std(arena["logret"], vw)
        thr = np.nanquantile(vol[:IS_STEPS], q)
        mom = np.zeros(T, dtype=bool)
        mom[mw:] = c[mw:] > c[:-mw]
        pos = ((vol < thr) & mom & ~np.isnan(vol)).astype(np.int8)

    else:  # pragma: no cover
        raise ValueError(family)
    return pos


def draw_params(family: str, rng: np.random.Generator) -> dict:
    if family == "trend":
        fast = int(rng.integers(3, 16))
        slow = int(rng.integers(max(fast + 5, 20), 61))
        return {"fast": fast, "slow": slow}
    if family == "breakout":
        return {"enter": int(rng.integers(10, 41)),
                "exit": int(rng.integers(5, 21))}
    if family == "meanrev":
        return {"window": int(rng.integers(10, 31)),
                "z_in": float(rng.uniform(1.0, 2.2)),
                "z_out": float(rng.uniform(0.0, 0.5))}
    if family == "voltimer":
        return {"vol_window": int(rng.integers(5, 21)),
                "quantile": float(rng.uniform(0.3, 0.7)),
                "mom": int(rng.integers(5, 31))}
    raise ValueError(family)


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------

def _runs(pos: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous pos==1 runs as (first_day, day_position_closes)."""
    runs = []
    entry = None
    for t in range(len(pos)):
        if pos[t] == 1 and entry is None:
            entry = t
        elif pos[t] == 0 and entry is not None:
            runs.append((entry, t))
            entry = None
    if entry is not None:
        runs.append((entry, len(pos) - 1))
    return runs


def apply_edge(arena: dict, pos: np.ndarray, p_edge: float,
               rng: np.random.Generator) -> np.ndarray:
    """Give a strategy a persistent, regime-conditional trade-level edge
    while preserving rule-like trade shapes (no single-day blips):

    - veto: each base trade that would lose money is skipped with
      probability p_edge if it starts on an eligible (high-volatility)
      day, p_edge / VETO_OFFREGIME_FACTOR otherwise;
    - insert: on eligible flat days, with probability
      INSERT_RATE * p_edge, open a 3-10 day hold, but only when that
      hold actually gained more than INSERT_MIN_GAIN.

    Applied over the full 560-day window, so the edge persists into the
    hidden continuation segment.
    """
    T = len(pos)
    c = arena["close"]
    out = pos.copy()

    # ---- veto losing base trades ----
    for entry, close_day in _runs(pos):
        u = rng.random()
        trade_ret = c[close_day] / c[entry] - 1.0
        p_veto = p_edge if arena["eligible"][entry] \
            else p_edge / VETO_OFFREGIME_FACTOR
        if trade_ret < 0.0 and u < p_veto:
            out[entry:close_day] = 0

    # ---- insert well-timed holds on eligible flat days ----
    p_ins = INSERT_RATE * p_edge
    u_day = rng.random(T)
    hold_draw = rng.integers(INSERT_HOLD_MIN, INSERT_HOLD_MAX + 1, size=T)
    t = VOL_WINDOW
    while t < T - INSERT_HOLD_MIN - 1:
        if out[t] == 1:
            t += 1
            continue
        if arena["eligible"][t] and u_day[t] < p_ins:
            h = int(hold_draw[t])
            end = min(t + h, T - 1)
            if end > t and c[end] / c[t] - 1.0 > INSERT_MIN_GAIN:
                out[t:end] = 1
                t = end + 1
                continue
        t += 1
    return out


def strategy_returns(arena: dict, pos: np.ndarray) -> np.ndarray:
    r = np.zeros(len(pos))
    r[1:] = pos[:-1] * arena["ret"][1:]
    return r


def sharpe(returns: np.ndarray) -> float:
    sd = returns.std()
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    val = returns.mean() / sd * ANNUALIZE
    return float(np.clip(val, -SHARPE_CLIP, SHARPE_CLIP))


def is_oos_sharpe(sr: np.ndarray) -> tuple[float, float]:
    return sharpe(sr[1:IS_STEPS]), sharpe(sr[IS_STEPS:])


def trades_from_positions(arena: dict, pos: np.ndarray) -> list[dict]:
    """In-sample trade log (steps 0..449 only)."""
    c = arena["close"]
    trades = []
    entry = None
    for t in range(IS_STEPS):
        if pos[t] == 1 and entry is None:
            entry = t
        elif pos[t] == 0 and entry is not None:
            trades.append({
                "entry_step": entry, "exit_step": t,
                "trade_return": c[t] / c[entry] - 1.0,
                "open_at_cutoff": 0,
            })
            entry = None
    if entry is not None:
        trades.append({
            "entry_step": entry, "exit_step": IS_STEPS - 1,
            "trade_return": c[IS_STEPS - 1] / c[entry] - 1.0,
            "open_at_cutoff": 1,
        })
    return trades


def acceptable_base(pos: np.ndarray) -> bool:
    is_pos = pos[:IS_STEPS]
    exposure = is_pos.mean()
    n_entries = int(((is_pos[1:] == 1) & (is_pos[:-1] == 0)).sum())
    return 0.05 <= exposure <= 0.95 and n_entries >= 6


def make_base_strategy(arena: dict, family: str, tag: str,
                       attempts: int = 200) -> tuple[dict, np.ndarray, bool]:
    """Draw parameters until the base rule trades enough (deterministic).
    Returns (params, positions, acceptable)."""
    for attempt in range(attempts):
        rng = _rng(f"{tag}:try{attempt}")
        params = draw_params(family, rng)
        pos = base_positions(arena, family, params)
        if acceptable_base(pos):
            return params, pos, True
    return params, pos, False


def make_acceptable_base(arena: dict, family: str,
                         tag: str) -> tuple[str, dict, np.ndarray]:
    """Like make_base_strategy but guaranteed acceptable: falls back to
    the other families (deterministic order) on arenas where a family's
    rule space never produces a tradeable configuration."""
    params, pos, ok = make_base_strategy(arena, family, tag)
    if ok:
        return family, params, pos
    for alt in FAMILIES:
        if alt == family:
            continue
        params, pos, ok = make_base_strategy(arena, alt, f"{tag}:alt:{alt}")
        if ok:
            return alt, params, pos
    raise RuntimeError(f"no acceptable base strategy for {tag}")


def _n_is_trades(pos: np.ndarray) -> int:
    is_pos = pos[:IS_STEPS]
    n = int(((is_pos[1:] == 1) & (is_pos[:-1] == 0)).sum())
    return n + int(is_pos[0] == 1)


def _is_winrate(arena: dict, pos: np.ndarray) -> float:
    trades = trades_from_positions(arena, pos)
    if not trades:
        return 0.5
    return float(np.mean([t["trade_return"] > 0 for t in trades]))


def simulate_population(arenas: dict[str, dict]) -> pd.DataFrame:
    """Simulate all strategies; returns one row per published strategy."""
    records = []
    for aid in sorted(arenas):
        arena = arenas[aid]

        # ---- informed strategies ----
        informed = []
        for i in range(N_INFORMED):
            family = FAMILIES[i % len(FAMILIES)]
            tag = f"{aid}:informed:{i}"
            family, params, base_pos = make_acceptable_base(
                arena, family, tag)
            erng = _rng(f"{tag}:edge")
            p_edge = float(erng.uniform(EDGE_MIN, EDGE_MAX))
            pos = apply_edge(arena, base_pos, p_edge, _rng(f"{tag}:peek"))
            retry = 0
            while _n_is_trades(pos) < 3 and retry < 30:
                retry += 1
                pos = apply_edge(arena, base_pos, p_edge,
                                 _rng(f"{tag}:peek:retry{retry}"))
            sr = strategy_returns(arena, pos)
            s_is, s_oos = is_oos_sharpe(sr)
            informed.append({
                "arena_id": aid, "family": family, "group": "informed",
                "skill": 1, "p_edge": p_edge, "is_sharpe": s_is,
                "is_winrate": _is_winrate(arena, pos),
                "oos_sharpe": s_oos, "pos": pos,
            })

        # ---- noise pool (for luck matching) ----
        pool = []
        for i in range(N_POOL):
            family = FAMILIES[i % len(FAMILIES)]
            tag = f"{aid}:pool:{i}"
            params, pos, ok = make_base_strategy(arena, family, tag)
            if not ok:
                continue  # this family never trades on this arena
            sr = strategy_returns(arena, pos)
            s_is, s_oos = is_oos_sharpe(sr)
            pool.append({
                "arena_id": aid, "family": family, "group": "luck",
                "skill": 0, "p_edge": 0.0, "is_sharpe": s_is,
                "is_winrate": _is_winrate(arena, pos),
                "oos_sharpe": s_oos, "pos": pos, "used": False,
            })

        # ---- luck selection: per family, nearest unused candidate in
        # (in-sample Sharpe, in-sample win rate) space, z-normalized ----
        all_sh = np.array([r["is_sharpe"] for r in informed + pool])
        all_wr = np.array([r["is_winrate"] for r in informed + pool])
        sh_sd = max(all_sh.std(), 1e-9)
        wr_sd = max(all_wr.std(), 1e-9)
        for inf in informed:
            candidates = [
                p for p in pool
                if p["family"] == inf["family"] and not p["used"]
            ]
            if not candidates:  # rare: fall back to any unused family
                candidates = [p for p in pool if not p["used"]]
            best = min(
                candidates,
                key=lambda p: (
                    ((p["is_sharpe"] - inf["is_sharpe"]) / sh_sd) ** 2
                    + ((p["is_winrate"] - inf["is_winrate"]) / wr_sd) ** 2
                ),
            )
            best["used"] = True

        luck = [p for p in pool if p["used"]]

        # ---- background noise (not matched) ----
        background = []
        brng = _rng(f"{aid}:bg-families")
        i = 0
        while len(background) < N_BACKGROUND:
            family = FAMILIES[int(brng.integers(0, len(FAMILIES)))]
            tag = f"{aid}:background:{i}"
            i += 1
            family, params, pos = make_acceptable_base(arena, family, tag)
            sr = strategy_returns(arena, pos)
            s_is, s_oos = is_oos_sharpe(sr)
            background.append({
                "arena_id": aid, "family": family, "group": "background",
                "skill": 0, "p_edge": 0.0, "is_sharpe": s_is,
                "is_winrate": _is_winrate(arena, pos),
                "oos_sharpe": s_oos, "pos": pos,
            })

        records.extend(informed + luck + background)

    df = pd.DataFrame(records)
    # global anonymized strategy ids, order-shuffled so ids carry no label
    idx = _rng("strategy-ids").permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    df["strategy_id"] = [f"S{k + 1:05d}" for k in range(len(df))]
    return df


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def write_outputs(arenas: dict[str, dict], pop: pd.DataFrame,
                  public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    # ---- arena split ----
    aids = sorted(arenas)
    split_rng = _rng("arena-split")
    train_arenas = set(
        np.array(aids)[split_rng.permutation(N_ARENAS)[:N_TRAIN_ARENAS]]
    )
    pop = pop.copy()
    pop["split"] = np.where(pop["arena_id"].isin(train_arenas),
                            "train", "test")

    # ---- arenas.csv (public in-sample bars) ----
    rows = []
    for aid in aids:
        a = arenas[aid]
        for t in range(IS_STEPS):
            rows.append((aid, t,
                         round(a["open"][t], 8), round(a["high"][t], 8),
                         round(a["low"][t], 8), round(a["close"][t], 8),
                         round(a["volume"][t], 8)))
    arenas_df = pd.DataFrame(
        rows, columns=["arena_id", "step", "open", "high", "low",
                       "close", "volume"],
    )
    arenas_df.to_csv(public / "arenas.csv", index=False)

    # ---- trades.csv (public, in-sample only, train + test strategies) ----
    trade_rows = []
    for rec in pop.itertuples():
        for tr in trades_from_positions(arenas[rec.arena_id], rec.pos):
            trade_rows.append((
                rec.strategy_id, tr["entry_step"], tr["exit_step"],
                round(tr["trade_return"], 8), tr["open_at_cutoff"],
            ))
    trades_df = pd.DataFrame(
        trade_rows,
        columns=["strategy_id", "entry_step", "exit_step",
                 "trade_return", "open_at_cutoff"],
    ).sort_values(["strategy_id", "entry_step"], kind="mergesort")
    missing = set(pop["strategy_id"]) - set(trades_df["strategy_id"])
    if missing:
        raise RuntimeError(
            f"{len(missing)} published strategies have no in-sample "
            f"trades (e.g. {sorted(missing)[:5]})")
    trades_df.to_csv(public / "trades.csv", index=False)

    # ---- train.csv / test.csv ----
    pop["oos_sharpe_r"] = pop["oos_sharpe"].round(4)
    train = pop[pop["split"] == "train"].sort_values("strategy_id")
    test = pop[pop["split"] == "test"].sort_values("strategy_id")

    train[["strategy_id", "arena_id", "family", "skill", "oos_sharpe_r"]] \
        .rename(columns={"oos_sharpe_r": "oos_sharpe"}) \
        .to_csv(public / "train.csv", index=False)
    test[["strategy_id", "arena_id", "family"]] \
        .to_csv(public / "test.csv", index=False)

    # ---- sample_submission.csv ----
    sample = test[["strategy_id"]].copy()
    sample["p_skill"] = 0.5
    sample["oos_sharpe"] = 0.0
    sample["select"] = 0
    sample.iloc[:K_SELECT, sample.columns.get_loc("select")] = 1
    sample.to_csv(public / "sample_submission.csv", index=False)

    # ---- private answers.csv ----
    terciles = pd.qcut(test["is_sharpe"], 3, labels=["low", "mid", "high"])
    answers = test[["strategy_id", "skill", "oos_sharpe_r",
                    "family", "arena_id"]].copy()
    answers = answers.rename(columns={"oos_sharpe_r": "oos_sharpe"})
    answers["is_sharpe_tercile"] = terciles.astype(str).values
    answers.to_csv(private / "answers.csv", index=False)

    n_tr, n_te = len(train), len(test)
    n_informed_te = int(test["skill"].sum())
    print(f"[ok] arenas.csv: {len(arenas_df)} rows "
          f"({N_ARENAS} arenas x {IS_STEPS} steps)")
    print(f"[ok] trades.csv: {len(trades_df)} rows")
    print(f"[ok] train.csv: {n_tr} strategies | test.csv: {n_te} "
          f"(informed in test: {n_informed_te})")
    print(f"[ok] sample_submission.csv: {n_te} rows, K={K_SELECT}")
    print(f"[ok] answers.csv: {n_te} rows -> {private}")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def _find_prices_csv(dataset_dir: Path) -> Path | None:
    root = Path(__file__).resolve().parent
    candidates = [
        dataset_dir / "prices.csv",
        dataset_dir / "raw" / "prices.csv",
        dataset_dir / "dataset" / "raw" / "prices.csv",
        root / "dataset" / "raw" / "prices.csv",
        root / "data" / "raw" / "prices.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_prepared(dataset_dir: Path) -> tuple[Path, Path] | None:
    root = Path(__file__).resolve().parent
    for base in (dataset_dir, root / "dataset"):
        pub, priv = base / "public", base / "private"
        if all((pub / f).exists() for f in PUBLIC_FILES) and \
                all((priv / f).exists() for f in PRIVATE_FILES):
            return pub, priv
    return None


def prepare(dataset_dir: Path, public_dir: Path, private_dir: Path) -> None:
    dataset_dir = Path(dataset_dir)
    public_dir = Path(public_dir)
    private_dir = Path(private_dir)

    prices_csv = _find_prices_csv(dataset_dir)

    if prices_csv is None:
        ready = _find_prepared(dataset_dir)
        if ready is not None:
            src_pub, src_priv = ready
            public_dir.mkdir(parents=True, exist_ok=True)
            private_dir.mkdir(parents=True, exist_ok=True)
            for f in PUBLIC_FILES:
                if (public_dir / f).resolve() != (src_pub / f).resolve():
                    shutil.copy2(src_pub / f, public_dir / f)
            for f in PRIVATE_FILES:
                if (private_dir / f).resolve() != (src_priv / f).resolve():
                    shutil.copy2(src_priv / f, private_dir / f)
            print(f"[ok] copied prepared files from {src_pub.parent} "
                  f"(raw prices.csv not found)")
            return
        # last resort: documented public download
        print("[info] raw prices.csv not found; attempting the documented "
              "public download via download_data.py ...")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import download_data  # noqa: PLC0415
        download_data.main()
        prices_csv = _find_prices_csv(dataset_dir)
        if prices_csv is None:
            raise FileNotFoundError(
                "prices.csv not found after download; cannot prepare."
            )

    print(f"[info] raw prices: {prices_csv}")
    prices = load_prices(prices_csv)
    arenas = build_arenas(prices)
    print(f"[ok] built {len(arenas)} arenas "
          f"({IS_STEPS} public + {OOS_STEPS} hidden steps each)")
    pop = simulate_population(arenas)
    print(f"[ok] simulated {len(pop)} published strategies")
    write_outputs(arenas, pop, public_dir, private_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    prepare(root / "dataset",
            root / "dataset" / "public",
            root / "dataset" / "private")
