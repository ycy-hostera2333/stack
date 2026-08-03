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
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder 平滑
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100 * (avg_gain > 0))


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    dif = ema(s, fast) - ema(s, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


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


def add_common(df: pd.DataFrame) -> pd.DataFrame:
    """给单只股票的日线补上常用指标列。df 需含 open/high/low/close/volume。

    这里只算「内置策略实际用得到」的一组。原因是性能：全市场扫描要对两千多只股票
    分别调用本函数，而单只股票的耗时几乎全部来自 pandas 的**每次调用开销**
    （每个 rolling/ewm 约 0.3-0.5ms），不是计算量本身。多算一个没人用的指标，
    全市场就要多花一秒多。实测把 MACD/布林/MA250 等移出后，扫描从 40s 降到 20s。

    需要更多指标就在策略里覆写 prepare()，先 super().prepare(df) 再调 add_extended(df)
    或自己加列——只有真正用到的策略才付这份开销。
    """
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    for n in (5, 10, 20, 60, 120):
        df[f"ma{n}"] = sma(c, n)
    df["vol_ma20"] = sma(v, 20)
    df["vol_ratio"] = v / df["vol_ma20"]

    df["rsi14"] = rsi(c, 14)
    df["atr14"] = atr(h, l, c, 14)
    df["atr_pct"] = df["atr14"] / c

    df["high20"] = rolling_high(h, 20)
    df["low20"] = rolling_low(l, 20)
    df["dd"] = drawdown(c)

    df["mom20"] = momentum(c, 20)
    df["mom60"] = momentum(c, 60)
    df["mom120"] = momentum(c, 120)
    df["vol20"] = volatility(c, 20)
    return df


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
