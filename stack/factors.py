"""因子研究：横截面因子库 + IC 评估 + 分层回测。

为什么先做这个而不是直接训模型：本地数据的**有效独立观测只有约 99 个月**
（2078 个交易日，但同一天所有股票被市场 beta 绑定，横截面上不独立）。
拿几十上百个特征去拟合 99 个观测，"最好的因子"几乎必然是噪声。
所以要先有一个能在建策略之前就廉价毙掉噪声因子的工具。

口径与回测引擎严格一致：
    t 日收盘算出因子 → t+1 日开盘买入 → t+1+N 日开盘卖出
因此前瞻收益用 open[t+1+N] / open[t+1] - 1，而不是常见的 close-to-close——
后者隐含"收盘价瞬间成交"，是拿不到的价格。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind
from .data import store, universe

REGISTRY: dict[str, "Factor"] = {}


@dataclass
class Factor:
    name: str
    label: str
    fn: Callable[[dict], pd.DataFrame]
    direction: int = 1      # +1 表示值越大越看多，-1 表示越小越看多
    note: str = ""


def factor(name: str, label: str, direction: int = 1, note: str = ""):
    def deco(fn):
        REGISTRY[name] = Factor(name, label, fn, direction, note)
        return fn
    return deco


# ------------------------------------------------------------------ 面板
def build_panel(codes: list[str] | None = None, start: str | None = None,
                end: str | None = None, top: int = 1200) -> dict:
    """构建横截面宽表 {字段: DataFrame(index=日期, columns=股票代码)}。

    因子几乎都是横截面运算（排序、分组、标准化），宽表可以整块向量化，
    比按股票逐个循环快一到两个数量级。
    """
    if codes is None:
        uni = universe.build(as_of=start)
        codes = uni["code"].tolist()[:top]
    raw = store.load_daily(codes=codes, start=start, end=end)
    if raw.empty:
        return {}

    # 负价历史（前复权累计分红超过当年股价）必须先截断，否则污染所有因子
    keep = []
    for _, g in raw.groupby("code", sort=False):
        keep.append(store.usable_history(g.sort_values("date")))
    raw = pd.concat(keep, ignore_index=True)

    P = {}
    for f in ("open", "high", "low", "close", "volume", "amount"):
        P[f] = raw.pivot(index="date", columns="code", values=f).astype("float32")
    P["pct_chg"] = raw.pivot(index="date", columns="code",
                             values="pct_chg").astype("float32")
    return P


def forward_return(P: dict, horizon: int) -> pd.DataFrame:
    """可执行的前瞻收益：t 日因子 → t+1 开盘买 → t+1+horizon 开盘卖。"""
    o = P["open"]
    return (o.shift(-1 - horizon) / o.shift(-1) - 1)


def tradable_mask(P: dict) -> pd.DataFrame:
    """t+1 日能否真正买进。剔除停牌和开盘涨停——信号再好也买不到。"""
    o, c = P["open"], P["close"]
    nxt_open = o.shift(-1)
    limit = pd.Series(
        [0.20 if str(x)[:3] in ("300", "301", "688", "689") else 0.10
         for x in o.columns], index=o.columns, dtype="float32")
    up = nxt_open >= c * (1 + limit) - 1e-6
    return nxt_open.notna() & c.notna() & ~up


# ------------------------------------------------------------------ 因子库
@factor("mom_20", "20日动量", note="短期动量，A股上常表现为反转")
def _f_mom20(P):
    return P["close"] / P["close"].shift(20) - 1


@factor("mom_60", "60日动量")
def _f_mom60(P):
    return P["close"] / P["close"].shift(60) - 1


@factor("mom_120_20", "12-1动量(120日剔除近20日)",
        note="学界标准做法，跳过最近一个月以规避短期反转")
def _f_mom120s(P):
    c = P["close"]
    return c.shift(20) / c.shift(140) - 1


@factor("reversal_5", "5日反转", direction=-1,
        note="短期跌得多的反而涨，方向为负")
def _f_rev5(P):
    return P["close"] / P["close"].shift(5) - 1


@factor("vol_20", "20日年化波动率", direction=-1, note="低波动异象")
def _f_vol20(P):
    return P["close"].pct_change().rolling(20).std() * np.sqrt(252)


@factor("max_ret_20", "20日内最大单日涨幅", direction=-1,
        note="彩票效应：博弈性强的股票长期跑输")
def _f_max20(P):
    return P["pct_chg"].rolling(20).max()


@factor("amihud", "Amihud非流动性", direction=-1,
        note="单位成交额推动的价格变动，越大越难交易")
def _f_amihud(P):
    r = P["close"].pct_change().abs()
    return (r / P["amount"].replace(0, np.nan)).rolling(20).mean() * 1e9


@factor("turnover_20", "20日均换手代理(成交额)", direction=-1,
        note="用成交额代理，规模/流动性因子")
def _f_to20(P):
    return np.log(P["amount"].rolling(20).mean().replace(0, np.nan))


@factor("ma60_dist", "偏离MA60幅度")
def _f_ma60(P):
    c = P["close"]
    return c / c.rolling(60).mean() - 1


@factor("vol_ratio", "量比(量/20日均量)")
def _f_vr(P):
    v = P["volume"]
    return v / v.rolling(20).mean().replace(0, np.nan)


@factor("rsi_14", "RSI14", direction=-1)
def _f_rsi(P):
    d = P["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


@factor("mom_vol", "动量/波动率", note="偏好走得稳的强势股，策略里用的打分")
def _f_momvol(P):
    c = P["close"]
    mom = c.shift(20) / c.shift(140) - 1
    vol = c.pct_change().rolling(20).std() * np.sqrt(252)
    return mom / vol.replace(0, np.nan)


# ------------------------------------------------------------------ 基本面因子
# 全部走 point-in-time：每个交易日只能看到**已过法定披露截止日**的报告。
# 详见 data/fundamental.py 的说明——按报告期对齐是最隐蔽的一类前视偏差。
def _fund(P: dict, field: str) -> pd.DataFrame:
    """取某个基本面字段的 PIT 宽表，与价格面板对齐。"""
    from .data import fundamental as fd
    close = P["close"]
    dates = close.index.strftime("%Y-%m-%d").tolist()
    panel = fd.as_panel(field, dates, list(close.columns))
    panel.index = close.index
    return panel


@factor("roe", "净资产收益率", note="盈利能力，PIT 对齐")
def _f_roe(P):
    return _fund(P, "roe")


@factor("gross_margin", "销售毛利率", note="定价能力/护城河代理")
def _f_gm(P):
    return _fund(P, "gross_margin")


@factor("profit_yoy", "净利润同比增长", note="成长性")
def _f_pyoy(P):
    return _fund(P, "profit_yoy")


@factor("revenue_yoy", "营收同比增长", note="成长性，比净利更难粉饰")
def _f_ryoy(P):
    return _fund(P, "revenue_yoy")


@factor("pb", "市净率", direction=-1,
        note="低估值溢价。用收盘价 / 每股净资产现算，两边都是 PIT 的")
def _f_pb(P):
    bps = _fund(P, "bps")
    return P["close"] / bps.where(bps > 0)


@factor("ep", "盈利收益率(EPS/价格)", note="市盈率的倒数，避免 PE 在亏损时符号翻转")
def _f_ep(P):
    # 用倒数而非 PE：EPS 为负时 PE 会变成"很小的负数"，排序上反而排到前面，
    # 是估值因子最常见的陷阱。EP 为负就是负，排序天然正确。
    eps = _fund(P, "eps")
    return eps / P["close"].replace(0, np.nan)


@factor("ocf_quality", "经营现金流/每股收益", direction=1,
        note="盈利质量：赚的钱有没有真的收回来")
def _f_ocfq(P):
    eps, ocf = _fund(P, "eps"), _fund(P, "ocfps")
    return ocf / eps.where(eps.abs() > 1e-6)


# ------------------------------------------------------------------ 评估
def _rank_ic(f: pd.DataFrame, r: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """逐日横截面 Rank IC（斯皮尔曼相关）。"""
    f = f.where(mask)
    r = r.where(mask)
    fr = f.rank(axis=1)
    rr = r.rank(axis=1)
    # 逐行相关：手算比 corrwith 快得多
    fc = fr.sub(fr.mean(axis=1), axis=0)
    rc = rr.sub(rr.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc ** 2).sum(axis=1) * (rc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    n = f.notna().sum(axis=1)
    return ic.where(n >= 30)          # 横截面样本太少的日子不计


def evaluate(name: str, P: dict, horizons=(1, 5, 10, 20, 60),
             quantiles: int = 5, main_h: int = 20) -> dict:
    """评估单个因子：IC、IC 衰减、分层收益、换手率。"""
    fac = REGISTRY[name]
    f = fac.fn(P) * fac.direction
    mask = tradable_mask(P)

    out = {"name": name, "label": fac.label, "direction": fac.direction,
           "note": fac.note, "ic": {}}

    for h in horizons:
        ic = _rank_ic(f, forward_return(P, h), mask).dropna()
        if len(ic) < 30:
            continue
        mean, sd = float(ic.mean()), float(ic.std(ddof=1))
        ir = mean / sd if sd > 0 else 0.0
        out["ic"][h] = {
            "mean": mean, "std": sd, "ir": ir,
            "t": ir * np.sqrt(len(ic)),        # 简化 t 值，未做自相关校正
            "pos_ratio": float((ic > 0).mean()),
            "n": int(len(ic)),
        }

    # ---- 分层：按因子值分 N 组，看收益是否单调 ----
    fw = forward_return(P, main_h)
    fm, rm = f.where(mask), fw.where(mask)
    q = fm.rank(axis=1, pct=True)
    layers = []
    for i in range(quantiles):
        lo, hi = i / quantiles, (i + 1) / quantiles
        sel = (q > lo) & (q <= hi) if i else (q >= 0) & (q <= hi)
        layers.append(float(rm.where(sel).mean(axis=1).mean()))
    out["layers"] = layers
    out["spread"] = layers[-1] - layers[0] if layers else None
    # 单调性：相邻层递增的比例
    inc = sum(1 for a, b in zip(layers, layers[1:]) if b > a)
    out["monotonic"] = inc / max(len(layers) - 1, 1)

    # ---- 头部超额：**这才是引擎实际能拿到的** ----
    # 引擎按 score 取前 N 名建仓，所以决定成败的是最高分组相对全池的超额，
    # 而不是 rank IC。IC 衡量整个横截面的单调关联，可能全部来自分布中段——
    # 实测 vol_20 的 IC 逐年 8/8 为正，头部超额却是负的，据此设计的策略
    # 选出来的股票平均跑输全池。用错判据比没有判据更危险。
    top = rm.where(q > 1 - 1 / quantiles).mean(axis=1)
    uni = rm.mean(axis=1)
    ex = (top - uni).dropna()
    out["top_excess"] = float(ex.mean()) if len(ex) else None
    if len(ex):
        by = ex.groupby(ex.index.year).mean()
        out["top_win_years"] = int((by > 0).sum())
        out["top_years"] = int(len(by))
        out["top_by_year"] = {str(k): round(float(v), 4) for k, v in by.items()}

    # ---- 换手：最高分位组的成分变化率，决定会不会被手续费吃光 ----
    topq = (q > 1 - 1 / quantiles)
    prev = topq.shift(1)
    both = topq & prev
    out["turnover"] = float(1 - (both.sum(axis=1) /
                                 topq.sum(axis=1).replace(0, np.nan)).mean())
    return out


def evaluate_all(P: dict, **kw) -> pd.DataFrame:
    rows = []
    for name in REGISTRY:
        try:
            r = evaluate(name, P, **kw)
        except Exception as e:
            rows.append({"因子": name, "错误": str(e)[:40]})
            continue
        ic20 = r["ic"].get(20, {})
        rows.append({
            "因子": name, "名称": r["label"],
            "头部超额": r.get("top_excess"),
            "头部赢年": (f"{r.get('top_win_years')}/{r.get('top_years')}"
                     if r.get("top_years") else None),
            "多空价差": r["spread"], "单调性": r["monotonic"],
            "IC均值": ic20.get("mean"), "换手率": r["turnover"],
        })
    df = pd.DataFrame(rows)
    # 按「头部超额」排序而不是 IC——引擎买的是头部
    if "头部超额" in df:
        df = df.sort_values("头部超额", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def correlation(P: dict, names: list[str] | None = None) -> pd.DataFrame:
    """因子间的横截面秩相关均值——防止「发现」一堆其实是同一个东西的因子。"""
    names = names or list(REGISTRY)
    mask = tradable_mask(P)
    ranks = {}
    for n in names:
        f = (REGISTRY[n].fn(P) * REGISTRY[n].direction).where(mask)
        ranks[n] = f.rank(axis=1)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, a in enumerate(names):
        for b in names[i:]:
            ra, rb = ranks[a], ranks[b]
            ac = ra.sub(ra.mean(axis=1), axis=0)
            bc = rb.sub(rb.mean(axis=1), axis=0)
            num = (ac * bc).sum(axis=1)
            den = np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
            v = float((num / den.replace(0, np.nan)).mean())
            out.loc[a, b] = out.loc[b, a] = v
    return out.astype(float)
