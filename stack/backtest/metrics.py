"""绩效指标。

看回测别只看总收益。判断一个策略能不能真上，重点在这几个：
  最大回撤 —— 你实际能不能扛住不割
  卡玛比率 —— 每承受 1 单位回撤换来多少年化
  交易次数 —— 太少则统计不显著，太多则被手续费吃光
  盈亏比 × 胜率 —— 决定策略的性格（高胜率小赚 vs 低胜率大赚）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def analytics(equity: pd.DataFrame, trades: list, initial_cash: float) -> dict:
    """给界面用的时序分析数据：回撤曲线、月度/年度收益、仓位暴露、收益分布。

    这些都在服务端算好再传，避免前端重复实现一遍统计逻辑——
    两处实现早晚会算出不一样的数字。
    """
    if equity is None or equity.empty:
        return {}

    eq = equity.copy()
    eq["dt"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("dt")
    v = eq["equity"]

    # ---- 回撤曲线 ----
    dd = (v / v.cummax() - 1).round(5)

    # ---- 月度收益（年 × 月 网格）----
    mv = v.resample("ME").last()
    mret = mv.pct_change()
    if len(mv):   # 首月要用初始资金作基数，否则第一格永远是空的
        mret.iloc[0] = mv.iloc[0] / initial_cash - 1
    monthly = {}
    for ts, r in mret.items():
        if pd.notna(r):
            monthly.setdefault(str(ts.year), {})[str(ts.month)] = round(float(r), 4)

    # ---- 年度收益，与基准对比 ----
    yearly = []
    yv = v.resample("YE").last()
    prev = initial_cash
    bench_series = None
    if "benchmark" in eq.columns and eq["benchmark"].notna().any():
        bench_series = eq["benchmark"].ffill().resample("YE").last()
    bprev = None
    for ts, val in yv.items():
        row = {"year": str(ts.year), "ret": round(float(val / prev - 1), 4)}
        if bench_series is not None and ts in bench_series.index:
            bv = float(bench_series.loc[ts])
            if bprev is None:
                first = eq["benchmark"].ffill().dropna()
                bprev = float(first.iloc[0]) if len(first) else bv
            row["bench"] = round(bv / bprev - 1, 4)
            bprev = bv
        yearly.append(row)
        prev = float(val)

    # ---- 仓位暴露：持仓市值占总资产比例 ----
    exposure = ((v - eq["cash"]) / v).clip(0, 1).round(4) if "cash" in eq else None

    # ---- 单笔收益分布（直方图分桶）----
    dist = []
    if trades:
        pcts = np.array([t.pnl_pct for t in trades], dtype=float)
        edges = np.array([-1, -.30, -.20, -.15, -.10, -.05, 0,
                          .05, .10, .15, .20, .30, .50, 10.0])
        cnt, _ = np.histogram(pcts, bins=edges)
        labels = ["<-30%", "-30~-20", "-20~-15", "-15~-10", "-10~-5", "-5~0",
                  "0~5%", "5~10", "10~15", "15~20", "20~30", "30~50", ">50%"]
        dist = [{"label": l, "count": int(c)} for l, c in zip(labels, cnt)]

    return {
        "drawdown": dd.tolist(),
        "exposure": exposure.tolist() if exposure is not None else [],
        "positions": eq["positions"].tolist() if "positions" in eq else [],
        "monthly": monthly,
        "yearly": yearly,
        "trade_dist": dist,
    }


def compute(equity: pd.DataFrame, trades: list, initial_cash: float) -> dict:
    if equity is None or equity.empty:
        return {"error": "无净值序列"}

    eq = equity["equity"].to_numpy(dtype=float)
    n = len(eq)
    total_return = eq[-1] / initial_cash - 1
    years = n / TRADING_DAYS
    cagr = (eq[-1] / initial_cash) ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else 0.0

    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    vol = rets.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(rets) > 1 else 0.0
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0

    downside = rets[rets < 0]
    dvol = downside.std(ddof=0) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 0.0
    sortino = (rets.mean() * TRADING_DAYS) / dvol if dvol > 0 else 0.0

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = float(dd.min())
    # 最长水下时间：从上一次创新高到再次创新高的最大间隔
    underwater, longest, cur = eq < peak, 0, 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    out = {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "max_drawdown_days": int(longest),
        "volatility": round(float(vol), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "final_equity": round(float(eq[-1]), 2),
        "days": n,
    }

    # ------------------------------------------------------------ 交易统计
    if trades:
        pnls = np.array([t.pnl for t in trades], dtype=float)
        pcts = np.array([t.pnl_pct for t in trades], dtype=float)
        holds = np.array([t.hold_days for t in trades], dtype=float)
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        gross_win, gross_loss = wins.sum(), -losses.sum()
        out.update({
            "trades": len(trades),
            "win_rate": round(len(wins) / len(trades), 4),
            "avg_return": round(float(pcts.mean()), 4),
            "avg_win": round(float(pcts[pnls > 0].mean()), 4) if len(wins) else 0.0,
            "avg_loss": round(float(pcts[pnls <= 0].mean()), 4) if len(losses) else 0.0,
            "profit_factor": round(float(gross_win / gross_loss), 3) if gross_loss > 0 else None,
            "avg_hold_days": round(float(holds.mean()), 1),
            "best_trade": round(float(pcts.max()), 4),
            "worst_trade": round(float(pcts.min()), 4),
        })
    else:
        out.update({"trades": 0, "win_rate": 0.0,
                    "note": "区间内没有产生任何交易，可能是过滤条件过严或数据不足"})

    # ------------------------------------------------------------ 对比基准
    if "benchmark" in equity.columns and equity["benchmark"].notna().any():
        bm = equity["benchmark"].ffill().to_numpy(dtype=float)
        valid = np.isfinite(bm)
        if valid.sum() > 1:
            bm = bm[valid]
            bm_ret = bm[-1] / bm[0] - 1
            bm_peak = np.maximum.accumulate(bm)
            out["benchmark_return"] = round(float(bm_ret), 4)
            out["benchmark_max_drawdown"] = round(float((bm / bm_peak - 1).min()), 4)
            out["excess_return"] = round(float(total_return - bm_ret), 4)

    return out
