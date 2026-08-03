"""股票池过滤。

选股之前先把"根本不该碰的票"剔掉，比在策略里打补丁干净得多：
ST 股（退市风险 + 5% 涨跌停）、次新股（无历史可回测、炒作剧烈）、
低流动性股（信号再好也买不进卖不出）、长期停牌股。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from . import store


@dataclass
class UniverseFilter:
    exclude_st: bool = True
    exclude_star: bool = False       # 科创板：20% 涨跌停 + 50万门槛
    exclude_chinext: bool = False    # 创业板：20% 涨跌停
    exclude_bj: bool = True          # 北交所：流动性普遍偏弱
    min_listed_days: int = 250       # 上市满一年才纳入
    min_amount: float = 5e7          # 20 日均成交额下限（元），默认 5000 万
    min_price: float = 2.0           # 剔除低价股，避免 1 分钱跳动带来的失真
    max_price: float = 1e9

    def board_blacklist(self) -> set[str]:
        bl = set()
        if self.exclude_star:
            bl.add("科创板")
        if self.exclude_chinext:
            bl.add("创业板")
        if self.exclude_bj:
            bl.add("北交所")
        bl.add("未知")
        return bl


def build(flt: UniverseFilter | None = None, as_of: str | None = None) -> pd.DataFrame:
    """返回通过过滤的股票，含 code/name/board/industry 及流动性统计。

    as_of 用于回测时按当时的数据判定，避免用今天的流动性去筛几年前的池子。
    """
    flt = flt or UniverseFilter()
    inst = store.load_instruments()
    if inst.empty:
        return inst

    as_of = as_of or datetime.now().strftime("%Y-%m-%d")
    inst = inst[~inst["board"].isin(flt.board_blacklist())]

    # ---- point-in-time：只保留在 as_of 当天真实存在的股票 ----
    # 这是修正幸存者偏差的关键一步。如果只用当前在市的股票建池子，
    # 等于每次都在"事后知道谁活下来了"的集合里选股，历史回测会系统性偏乐观。
    # 反过来也要防止用未来信息：as_of 之后才上市的公司当时并不存在。
    if "delisted_date" in inst.columns:
        dl = pd.to_datetime(inst["delisted_date"], errors="coerce")
        # 退市日晚于 as_of 的，在当时仍在交易，必须留在池子里
        inst = inst[dl.isna() | (dl > pd.Timestamp(as_of))]

    if flt.exclude_st:
        inst = inst[inst["is_st"] == 0]

    if flt.min_listed_days > 0:
        listed = pd.to_datetime(inst["listed_date"], errors="coerce")
        age = (pd.Timestamp(as_of) - listed).dt.days
        inst = inst[age >= flt.min_listed_days]

    # 流动性与价格用截止日前 20 个交易日的实际成交衡量
    days = store.trading_days(end=as_of)
    if not days:
        return inst.assign(avg_amount=pd.NA, last_close=pd.NA)
    window_start = days[-min(20, len(days))]

    recent = store.load_daily(codes=inst["code"].tolist(),
                              start=window_start, end=as_of)
    if recent.empty:
        return inst.assign(avg_amount=pd.NA, last_close=pd.NA)

    agg = recent.groupby("code").agg(
        avg_amount=("amount", "mean"),
        last_close=("close", "last"),
        bars=("close", "size"),
    ).reset_index()

    out = inst.merge(agg, on="code", how="inner")
    out = out[
        (out["avg_amount"] >= flt.min_amount)
        & (out["last_close"].between(flt.min_price, flt.max_price))
        & (out["bars"] >= 15)          # 窗口内至少有 15 个交易日，剔除长期停牌
    ]
    return out.sort_values("avg_amount", ascending=False).reset_index(drop=True)


def summary(flt: UniverseFilter | None = None) -> dict:
    """给界面用的股票池统计。"""
    inst = store.load_instruments()
    uni = build(flt)
    by_board = uni["board"].value_counts().to_dict() if not uni.empty else {}
    return {
        "total_listed": int(len(inst)),
        "in_universe": int(len(uni)),
        "by_board": by_board,
    }
