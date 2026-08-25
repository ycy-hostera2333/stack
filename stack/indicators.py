"""技术指标，全部向量化，输入输出都是按单只股票时间升序的 DataFrame/Series。

约定：所有指标只用到当日及之前的数据（不含未来函数）。回测里再统一延后一天成交。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    # diff/clip/fillna 都走 numpy：全市场扫描要按股票逐只调用，
    # 每个 pandas 小操作 0.1-0.2ms 的固定开销乘以三千多只就很可观。
    # ewm 保留 pandas 版（Wilder 平滑，自己写反而慢）。
    v = np.asarray(s, dtype="float64")
    d = np.empty_like(v)
    d[0] = np.nan
    d[1:] = v[1:] - v[:-1]
    idx = s.index
    # Wilder 平滑
    avg_gain = pd.Series(np.clip(d, 0.0, None), index=idx).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = pd.Series(-np.clip(d, None, 0.0), index=idx).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean()
    ag, al = avg_gain.to_numpy(), avg_loss.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100.0 - 100.0 / (1.0 + ag / np.where(al == 0, np.nan, al))
    # 无涨跌（rs 为 NaN）或预热未满时：只涨记 100，其余记 0。
    # 这是原实现 .fillna(100 * (avg_gain > 0)) 的语义，逐位对齐保留。
    bad = np.isnan(out)
    out[bad] = np.where(ag[bad] > 0, 100.0, 0.0)
    return pd.Series(out, index=idx, name=s.name)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    dif = ema(s, fast) - ema(s, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # 三列拼成 DataFrame 再取行最大值，单只股票就要 0.47ms——全市场是 1.6s。
    # np.fmax 与 DataFrame.max(axis=1) 的缺失值语义一致（忽略 NaN，全 NaN 才是 NaN），
    # 首根 K 线没有前收，正好靠这一点退化成 high-low。
    hi = np.asarray(high, dtype="float64")
    lo = np.asarray(low, dtype="float64")
    cl = np.asarray(close, dtype="float64")
    prev = np.empty_like(cl)
    prev[0] = np.nan
    prev[1:] = cl[:-1]
    tr = np.fmax(np.fmax(hi - lo, np.abs(hi - prev)), np.abs(lo - prev))
    return pd.Series(tr, index=close.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def rolling_high(s: pd.Series, n: int) -> pd.Series:
    """过去 n 日（含当日）最高值。"""
    return s.rolling(n, min_periods=n).max()


def rolling_low(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()


def drawdown(s: pd.Series) -> pd.Series:
    """相对历史最高点的回撤（负值）。"""
    return s / s.cummax() - 1


def momentum(s: pd.Series, n: int) -> pd.Series:
    """n 日涨幅。"""
    return s / s.shift(n) - 1


def volatility(s: pd.Series, n: int = 20) -> pd.Series:
    """n 日收益率年化波动率。"""
    return s.pct_change().rolling(n, min_periods=n).std(ddof=0) * np.sqrt(252)


def slope(s: pd.Series, n: int) -> pd.Series:
    """n 日线性回归斜率，按均值归一化，衡量趋势强度。"""
    x = np.arange(n)
    x_c = x - x.mean()
    denom = (x_c ** 2).sum()

    def _fit(win: np.ndarray) -> float:
        return float((x_c * (win - win.mean())).sum() / denom / max(win.mean(), 1e-9))

    return s.rolling(n, min_periods=n).apply(_fit, raw=True)


# add_common 会产出的列，顺序即建列顺序。
COMMON_COLUMNS = (
    "ma5", "ma10", "ma20", "ma60", "ma120",
    "vol_ma20", "vol_ratio", "rsi14", "atr14", "atr_pct",
    "high20", "low20", "dd", "mom20", "mom60", "mom120", "vol20",
)


def add_common(df: pd.DataFrame, only=None) -> pd.DataFrame:
    """给单只股票的日线补上常用指标列。df 需含 open/high/low/close/volume。

    这里只算「内置策略实际用得到」的一组。原因是性能：全市场扫描要对两千多只股票
    分别调用本函数，而单只股票的耗时几乎全部来自 pandas 的**每次调用开销**
    （每个 rolling/ewm 约 0.3-0.5ms），不是计算量本身。多算一个没人用的指标，
    全市场就要多花一秒多。实测把 MACD/布林/MA250 等移出后，扫描从 40s 降到 20s。

    only 进一步收窄到指定的几列（策略通过 Strategy.indicators 声明）。
    依赖会自动带上：算 vol_ratio 时顺带算 vol_ma20，算 atr_pct 时顺带算 atr14，
    但只有被点名的列会出现在结果里。传 None（默认）等于全部，行为与从前一致。

    需要更多指标就在策略里覆写 prepare()，先 super().prepare(df) 再调 add_extended(df)
    或自己加列——只有真正用到的策略才付这份开销。
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    cache: dict[str, pd.Series] = {}

    def build(name: str) -> pd.Series:
        if name in cache:
            return cache[name]
        if name.startswith("ma") and name[2:].isdigit():
            s = sma(c, int(name[2:]))
        elif name.startswith("mom") and name[3:].isdigit():
            s = momentum(c, int(name[3:]))
        elif name == "vol_ma20":
            s = sma(v, 20)
        elif name == "vol_ratio":
            s = v / build("vol_ma20")
        elif name == "rsi14":
            s = rsi(c, 14)
        elif name == "atr14":
            s = atr(h, l, c, 14)
        elif name == "atr_pct":
            s = build("atr14") / c
        elif name == "high20":
            s = rolling_high(h, 20)
        elif name == "low20":
            s = rolling_low(l, 20)
        elif name == "dd":
            s = drawdown(c)
        elif name == "vol20":
            s = volatility(c, 20)
        else:                      # 只可能是调用方拼错了列名，必须当场报错
            raise KeyError(f"add_common 不认识的指标列 {name!r}，"
                           f"可用：{', '.join(COMMON_COLUMNS)}")
        cache[name] = s
        return s

    if only is None:
        names = COMMON_COLUMNS
    else:
        want = set(only)
        unknown = want - set(COMMON_COLUMNS)
        if unknown:
            raise KeyError(f"add_common 不认识的指标列 {sorted(unknown)}，"
                           f"可用：{', '.join(COMMON_COLUMNS)}")
        names = tuple(n for n in COMMON_COLUMNS if n in want)

    cols = {n: build(n) for n in names}
    if not cols:
        return df
    # 一次拼接，而不是逐列赋值：每次 df[col] = ... 都要重建一遍块管理器，
    # 单只股票看不出来，全市场两千多只乘十几列就很可观（实测 6.76ms -> 2.36ms）。
    # 这里返回的是新对象，调用方的 df 不受影响（原先靠 df.copy() 保证，同样成立）。
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def add_extended(df: pd.DataFrame) -> pd.DataFrame:
    """补充指标：MACD、布林带、MA250、60 日高点、5 日量均。

    默认不算——见 add_common 的说明。在策略里按需调用：

        def prepare(self, df):
            return ind.add_extended(super().prepare(df))
    """
    df = df.copy()
    c, h, v = df["close"], df["high"], df["volume"]
    df["ma250"] = sma(c, 250)
    df["vol_ma5"] = sma(v, 5)
    df["dif"], df["dea"], df["macd_hist"] = macd(c)
    df["boll_low"], df["boll_mid"], df["boll_up"] = bollinger(c, 20, 2.0)
    df["high60"] = rolling_high(h, 60)
    return df
