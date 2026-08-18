"""策略基类与注册表。

一个策略要回答三件事：
  entry(df) -> 今天是否给出买入信号
  exit(df)  -> 今天是否给出卖出信号
  score(df) -> 同日候选过多时的优先级（越大越优先）

所有函数返回与 df 等长的 Series，逐日对齐。信号只允许使用当日收盘及之前的信息，
回测引擎会统一按"信号日收盘产生、次日开盘成交"来撮合，避免未来函数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Type

import numpy as np
import pandas as pd

from .. import indicators as ind

REGISTRY: dict[str, Type["Strategy"]] = {}


def register(cls: Type["Strategy"]) -> Type["Strategy"]:
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class Strategy:
    """策略基类。子类覆写 name/label/defaults 和 entry/exit。"""

    name: str = "base"
    label: str = "基类"
    description: str = ""
    defaults: dict = field(default_factory=dict)
    # 可选：给界面用的参数说明 {参数名: {label, min, max, step, options, hint}}
    param_meta: dict = field(default_factory=dict)

    def __init__(self, **params):
        merged = {**self.defaults}
        for k, v in params.items():
            if k not in self.defaults:
                raise ValueError(
                    f"策略 {self.name} 没有参数 {k!r}，可用：{sorted(self.defaults)}")
            # 按默认值的类型转换，避免界面传来的字符串导致比较出错
            ref = self.defaults[k]
            try:
                if isinstance(ref, bool):
                    v = bool(v)
                elif isinstance(ref, int) and not isinstance(v, bool):
                    v = int(float(v))
                elif isinstance(ref, float):
                    v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"参数 {k} 的值 {v!r} 无法转换成 {type(ref).__name__}")
            merged[k] = v
        self.params = merged
        self.validate()

    def validate(self) -> None:
        """参数合法性校验。子类覆写；非法时抛 ValueError，界面会直接显示这条消息。

        务必在这里挡住无意义的组合（比如快线周期 >= 慢线周期）——否则策略会安静地
        跑出一个永远不触发的信号，看起来像"市场没机会"，实际是参数配错了。
        """
        return None

    # -------------------------------------------------------------- 钩子
    def market_regime(self, dates: list[str]) -> pd.Series | None:
        """大盘择时：返回与 dates 等长的布尔序列，False 表示当日禁止开新仓。

        单只股票的 entry/exit 看不到指数，所以这个判断必须在策略层单独给出，
        由引擎统一施加到所有买入信号上。返回 None 表示不做择时（永远允许开仓）。

        注意只能用当日及之前的指数数据——引擎会用第 t-1 日的值来决定第 t 日能否买入。
        """
        return None

    def warmup_bars(self) -> int:
        """算出有效信号所需的最少 K 线根数，引擎据此决定预加载多少历史数据。

        参数可调的策略必须覆写这个方法。否则把慢线设成 250 时，预热窗口不够长，
        均线全是 NaN，信号恒为 False——表现为"市场没机会"，实际是数据没喂够，
        这种静默失败比报错难查得多。
        """
        return 250

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """补指标列。默认加通用指标；策略需要特殊指标时覆写并调用 super()。"""
        return ind.add_common(df)

    def entry(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def exit(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # 多因子打分：声明「用哪些列、各占多少权重」，引擎会把每一列**按日期逐列**
    # 转成横截面百分位排名，再加权求和。
    #
    # 为什么必须由引擎做：score() 只看得到单只股票的时间序列。在里面写
    # `df["x"].rank(pct=True)` 排的是**时间**维度，而且用到了该股票的全部历史
    # 包括未来——既不是横截面，还引入前视偏差。只有到了引擎手里才有完整的
    # (股票 × 日期) 矩阵，那是唯一能正确做横截面归一的地方。
    #
    # 例：score_fields = [("f_rev_yoy", 1.0), ("f_bp", 1.0)]
    # 留空则沿用 score() 的返回值，按原始值排序（单因子够用）。
    score_fields: list = field(default_factory=list)

    def score(self, df: pd.DataFrame) -> pd.Series:
        """默认用 60 日动量排序，让强势股优先。"""
        return df.get("mom60", pd.Series(0.0, index=df.index)).fillna(-9.9)

    def reason(self, row: pd.Series, action: str) -> str:
        """给出人类可读的信号理由，界面上直接展示。"""
        return f"{self.label} {action}"

    # -------------------------------------------------------------- 元信息
    @classmethod
    def info(cls) -> dict:
        return {
            "name": cls.name,
            "label": cls.label,
            "description": cls.description,
            "defaults": cls.defaults,
            "param_meta": getattr(cls, "param_meta", {}),
        }


def blend_score_fields(values: dict[str, dict[str, float]],
                       fields: list[tuple[str, float]]) -> dict[str, float]:
    """把 {代码: {字段: 值}} 按字段做横截面百分位归一后加权合成，返回 {代码: 分数}。

    打分型策略（声明了 score_fields 的）自己的 score() 只是返回 0 的占位符——
    在策略内部只看得到单只股票的时序，在那里排序排的是时间维度、还会用到未来数据。
    真正的合成必须在拿齐当日全部候选之后做，也就是回测引擎、模拟盘、每日信号
    这三个地方各做一次。

    漏掉任何一处都不会报错：候选全是 0 分，"按打分取前 N 名"静默退化成
    "按代码序取前 N 名"。模拟盘和每日信号都曾经漏掉过。

    口径与引擎的 _cross_rank 一致：有效值归一到 [0,1]，缺失记 0 分。
    """
    codes = list(values)
    if not codes:
        return {}
    total = pd.Series(0.0, index=codes, dtype="float64")
    wsum = 0.0
    for col, w in fields:
        v = pd.Series({c: values[c].get(col, float("nan")) for c in codes},
                      dtype="float64")
        pct = (v.rank(method="average") - 1) / max(v.notna().sum() - 1, 1)
        total += w * pct.fillna(0.0)
        wsum += w
    if wsum:
        total /= wsum
    return {c: float(total[c]) for c in codes}


def get_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"未知策略 {name!r}，可用：{sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def all_strategies() -> list[dict]:
    return [cls.info() for cls in REGISTRY.values()]


# ------------------------------------------------------------------ 工具
def safe(cond) -> pd.Series:
    """把可能含 NaN 的条件转成干净的布尔 Series。"""
    return pd.Series(cond).fillna(False).astype(bool)


def cross_above(a: pd.Series, b: pd.Series) -> pd.Series:
    return safe((a > b) & (a.shift(1) <= b.shift(1)))


def cross_below(a: pd.Series, b: pd.Series) -> pd.Series:
    return safe((a < b) & (a.shift(1) >= b.shift(1)))


def index_last_date(symbol: str = "IDX000300") -> str | None:
    """本地该指数的最新日期，用于检查择时判断所依据的数据是否陈旧。"""
    from ..data import store
    df = store.load_daily(codes=[symbol])
    if df.empty:
        return None
    return df["date"].max().strftime("%Y-%m-%d")


def hysteresis(flags: pd.Series, confirm: int) -> pd.Series:
    """状态翻转需连续 confirm 天确认，抑制在阈值附近的反复抖动。

    二元择时的通病：指数在均线附近来回穿越时，全进全出会被反复打脸。
    实测 2024 年该策略因此空仓 145 天，错过了指数 +16.5% 的一整轮上涨。

    只使用当日及之前的值，不含未来信息。
    """
    if confirm <= 1:
        return flags
    v = flags.to_numpy(dtype=bool)
    out = np.empty_like(v)
    state = bool(v[0])
    streak = 0
    for i, cur in enumerate(v):
        if cur == state:
            streak = 0
        else:
            streak += 1
            if streak >= confirm:
                state = bool(cur)
                streak = 0
        out[i] = state
    return pd.Series(out, index=flags.index)


def index_above_ma(dates: list[str], symbol: str = "IDX000300",
                   n: int = 200) -> pd.Series | None:
    """指数收盘价是否站上 n 日均线，按 dates 对齐。本地没有该指数则返回 None。

    均线在完整的指数历史上计算后再对齐到回测区间，否则区间开头的 n 天会因为
    预热不足而全是 NaN——那会让策略在回测开头白白空仓几个月。

    对齐时会 ffill：指数偶尔缺一两天是正常的。但如果指数整体没跟上个股数据
    （比如只同步了个股没同步指数），ffill 会让择时**静默地用几天前的旧值**做判断。
    择时是本策略影响最大的开关，所以调用方应当用 index_last_date() 检查新鲜度。
    """
    from ..data import store
    df = store.load_daily(codes=[symbol])
    if df.empty or len(df) < n + 5:
        return None
    df = df.sort_values("date")
    s = pd.Series(df["close"].to_numpy(),
                  index=df["date"].dt.strftime("%Y-%m-%d"))
    above = s > s.rolling(n, min_periods=n).mean()
    return above.reindex(dates).ffill().fillna(False).astype(bool)
