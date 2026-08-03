"""每日信号生成。

回测验证过的策略，在最新一根 K 线上跑一遍，得到：
  买入候选 —— 按策略 score 排序，附带买入理由和参考仓位
  卖出提醒 —— 只针对你在 positions 表里登记的实际持仓

注意信号是**收盘后**产生、**次日**执行的，和回测口径一致。盘中跑出来的信号
用的是未走完的 K 线，会失真——所以默认只用已收盘的最后一个交易日。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .config import LOT_SIZE
from .data import source, store, universe
from .strategies import base as base_mod
from .strategies.base import Strategy


@dataclass
class SignalConfig:
    max_candidates: int = 10
    portfolio_value: float = 200_000.0
    max_positions: int = 5
    allow_partial_bar: bool = False   # 是否允许用尚未走完的当日 K 线


def _latest_usable_date(allow_partial: bool = False) -> tuple[str | None, bool]:
    """返回 (可用的最后交易日, 是否跳过了当日未完成的 K 线)。

    盘中同步会把没走完的当日 K 线写进库里：成交量只有半天的量，量比会被系统性压低，
    所有带量能条件的策略都会给不出信号。所以默认只用**已收盘**的交易日。
    """
    days = store.trading_days()
    if not days:
        return None, False
    last = days[-1]
    if allow_partial:
        return last, False
    if last == datetime.now().strftime("%Y-%m-%d") and not source.market_closed_today():
        return (days[-2] if len(days) >= 2 else None), True
    return last, False


def generate(strategy: Strategy, flt: universe.UniverseFilter | None = None,
             cfg: SignalConfig | None = None, as_of: str | None = None) -> dict:
    """跑一遍策略，返回买入候选与持仓卖出提醒。"""
    cfg = cfg or SignalConfig()
    skipped_partial = False
    if as_of is None:
        as_of, skipped_partial = _latest_usable_date(cfg.allow_partial_bar)
    if as_of is None:
        return {"error": "本地还没有行情数据，请先运行数据同步"}

    uni = universe.build(flt, as_of=as_of)
    if uni.empty:
        return {"error": "股票池为空，请放宽过滤条件或先同步行情"}

    name_map = dict(zip(uni["code"], uni["name"]))
    held = store.list_positions()
    held_codes = set(held["code"].astype(str)) if not held.empty else set()

    # 持仓可能已经掉出股票池（比如被 ST 了），但仍需要给出卖出提醒
    codes = sorted(set(uni["code"]) | held_codes)
    # 只加载够算指标的窗口，不要拉全部历史（全市场几百万行，白等好几分钟）。
    # 窗口长度问策略要：参数可调的策略把慢线设到 250 时，需要的历史远超默认值。
    need_days = max(600, int(strategy.warmup_bars() * 1.5) + 60)
    load_start = (pd.Timestamp(as_of) - pd.Timedelta(days=need_days)).strftime("%Y-%m-%d")
    raw = store.load_daily(codes=codes, start=load_start, end=as_of)
    min_bars = max(130, int(strategy.warmup_bars() * 0.9))
    if raw.empty:
        return {"error": "没有可用于计算的行情数据"}

    # 大盘择时：关闭时只报卖出提醒，不给买入候选——与回测口径一致
    days = store.trading_days(end=as_of)
    reg = strategy.market_regime(days)
    regime_on = True if reg is None else bool(reg.iloc[-1])

    # 择时依据的指数数据必须跟得上，否则会拿几天前的旧行情决定今天开不开仓
    regime_stale = None
    if reg is not None:
        idx_last = base_mod.index_last_date("IDX000300")
        if idx_last is None or idx_last < as_of:
            regime_stale = idx_last

    buys, sells, errors = [], [], 0
    for code, g in raw.groupby("code", sort=False):
        g = store.usable_history(g.sort_values("date"))
        if len(g) < min_bars or g["date"].iloc[-1].strftime("%Y-%m-%d") != as_of:
            continue                      # 数据不足或当日停牌
        try:
            d = strategy.prepare(g)
            row = d.iloc[-1]
            if regime_on and bool(strategy.entry(d).iloc[-1]) and code not in held_codes:
                buys.append({
                    "code": code,
                    "name": name_map.get(code, code),
                    "price": round(float(row["close"]), 2),
                    "score": float(strategy.score(d).iloc[-1]),
                    "reason": strategy.reason(row, "BUY"),
                    "ma20": round(float(row.get("ma20", float("nan"))), 2),
                    "rsi14": round(float(row.get("rsi14", float("nan"))), 1),
                    "vol_ratio": round(float(row.get("vol_ratio", float("nan"))), 2),
                    "atr_pct": round(float(row.get("atr_pct", float("nan"))), 4),
                })
            if code in held_codes and bool(strategy.exit(d).iloc[-1]):
                sells.append({
                    "code": code,
                    "name": name_map.get(code, code),
                    "price": round(float(row["close"]), 2),
                    "reason": strategy.reason(row, "SELL"),
                })
        except Exception:
            errors += 1
            continue

    buys.sort(key=lambda x: -x["score"])
    buys = buys[: cfg.max_candidates]

    # 给出参考手数，省得自己按计算器
    budget = cfg.portfolio_value / max(cfg.max_positions, 1)
    for b in buys:
        shares = int(budget / b["price"] // LOT_SIZE) * LOT_SIZE
        b["suggest_shares"] = shares
        b["suggest_amount"] = round(shares * b["price"], 2)
        b["score"] = round(b["score"], 4)
        # 一手都买不起时必须说清楚。高价股（如 400 元/股需 4 万一手）在每仓预算
        # 之下会向下取整成 0 手——直接显示"建议 0 股"等于给了个无法执行的建议。
        b["affordable"] = shares > 0
        b["min_amount"] = round(LOT_SIZE * b["price"], 2)
        if shares == 0:
            b["note"] = (f"每仓预算 {budget:,.0f} 元买不起一手"
                         f"（1 手 = {LOT_SIZE} 股 × {b['price']:.2f} "
                         f"= {b['min_amount']:,.0f} 元）")

    return {
        "as_of": as_of,
        "strategy": strategy.name,
        "strategy_label": strategy.label,
        "params": strategy.params,
        "universe_size": int(len(uni)),
        "buys": buys,
        "sells": sells,
        "errors": errors,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skipped_partial_bar": skipped_partial,
        "regime_on": regime_on,
        "regime_note": (
            (f"⚠ 大盘择时依据的沪深300 数据只到 {regime_stale or '（缺失）'}，"
             f"晚于当前的 {as_of}，择时判断用的是旧行情。请先跑一次"
             f"「数据管理 → 增量同步」再看信号。" if regime_stale else "")
            + ("" if regime_on else
               "大盘择时判定为空头，本策略今日不开新仓——"
               "买入候选为空是策略的设计行为，不是没扫到。")),
        "note": ("信号基于收盘价产生，参考次日开盘执行——与回测口径一致。"
                 + ("　今日尚未收盘，当日 K 线不完整（成交量只有半天的量，"
                    "会把量比压低到几乎所有量能条件都不成立），"
                    f"因此本次用的是上一个已收盘交易日 {as_of} 的数据。"
                    if skipped_partial else "")),
    }


def persist(result: dict) -> int:
    """把信号写进 signal_log，方便日后回看"当时为什么给了这个信号"。"""
    if "error" in result:
        return 0
    rows = []
    for action, key in (("BUY", "buys"), ("SELL", "sells")):
        for item in result.get(key, []):
            rows.append({
                "date": result["as_of"], "strategy": result["strategy"],
                "code": item["code"], "name": item["name"], "action": action,
                "price": item["price"], "reason": item["reason"],
            })
    return store.log_signals(rows)
