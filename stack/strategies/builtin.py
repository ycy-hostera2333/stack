"""内置策略。都是公开的经典逻辑，重点在于按 A 股特点做了约束（避开 ST、要求流动性、
用 ATR 止损而非固定百分比、趋势过滤放在第一位）。

这些是研究起点，不是投资建议——先用回测面板在你自己的时间段上验证再说。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import (Strategy, cross_above, cross_below, hysteresis,
                   index_above_ma, register, safe)


@register
class MaTrend(Strategy):
    name = "ma_trend"
    label = "均线多头突破"
    description = (
        "要求均线多头排列（MA5>MA20>MA60 且 MA60 向上），在放量突破 20 日新高时买入，"
        "跌破 MA20 时卖出。典型的趋势跟踪：牛市吃大段、震荡市反复止损，"
        "建议配合「跟踪止损 ×ATR」一起用（推荐 2.5）。"
    )
    defaults = {
        "vol_ratio_min": 1.2,   # 突破日量能需达到 20 日均量的倍数
        "ma_fast": 5, "ma_mid": 20, "ma_slow": 60,
    }
    param_meta = {
        "vol_ratio_min": {"label": "量比下限", "min": 0, "max": 10, "step": 0.1,
                          "hint": "突破日成交量需达到 20 日均量的多少倍"},
        "ma_fast": {"label": "快线周期", "min": 2, "max": 60, "step": 1},
        "ma_mid": {"label": "中线周期", "min": 5, "max": 120, "step": 5,
                   "hint": "同时也是离场均线：跌破它就卖"},
        "ma_slow": {"label": "慢线周期", "min": 10, "max": 250, "step": 10},
    }

    def entry(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        f, m, s = f"ma{p['ma_fast']}", f"ma{p['ma_mid']}", f"ma{p['ma_slow']}"
        aligned = (df[f] > df[m]) & (df[m] > df[s])
        slow_up = df[s] > df[s].shift(5)
        # 突破前一日的 20 日高点，用 shift(1) 保证不是拿当日高点比自己
        breakout = df["close"] > df["high20"].shift(1)
        volume_ok = df["vol_ratio"] >= p["vol_ratio_min"]
        return safe(aligned & slow_up & breakout & volume_ok)

    def exit(self, df: pd.DataFrame) -> pd.Series:
        # 只给结构性离场信号。ATR 跟踪止损依赖入场点，交给回测引擎的
        # trail_stop_atr 处理——向量化的信号函数不知道持仓是哪天建的。
        p = self.params
        return cross_below(df["close"], df[f"ma{p['ma_mid']}"])

    def reason(self, row: pd.Series, action: str) -> str:
        if action == "BUY":
            return (f"多头排列突破20日高 {row['high20']:.2f}，"
                    f"量比 {row['vol_ratio']:.1f}，MA20 {row['ma20']:.2f}")
        return f"跌破 MA20 {row['ma20']:.2f}"


@register
class MaCross(Strategy):
    name = "ma_cross"
    label = "双均线交叉"
    description = (
        "最经典的择时规则：快线上穿慢线（金叉）买入，下穿（死叉）卖出。"
        "两条均线的周期都可以自己调——短周期组合（如 5/20）信号多、噪音大，"
        "长周期组合（如 20/60）信号少、跟得慢但抗震荡。"
        "建议先用它跑几组参数，看看哪段周期在你关心的时间区间上站得住。"
    )
    defaults = {
        "fast": 10,          # 快线周期
        "slow": 30,          # 慢线周期
        "ma_type": "sma",    # sma=简单均线, ema=指数均线
        "confirm_pct": 0.0,  # 金叉需快线高出慢线的幅度(%)，0 为不要求
        "trend_filter": 0,   # >0 时要求收盘价站上该周期均线才允许买入
    }
    param_meta = {
        "fast": {"label": "快线周期", "min": 2, "max": 250, "step": 1},
        "slow": {"label": "慢线周期", "min": 3, "max": 500, "step": 1},
        "ma_type": {"label": "均线类型", "options": ["sma", "ema"]},
        "confirm_pct": {"label": "金叉确认幅度 %", "min": 0, "max": 20, "step": 0.5,
                        "hint": "要求快线高出慢线这么多才算有效金叉，用来过滤反复缠绕"},
        "trend_filter": {"label": "趋势过滤均线", "min": 0, "max": 500, "step": 10,
                         "hint": "0=关闭。设 120 表示只在收盘价站上 MA120 时才买"},
    }

    def validate(self) -> None:
        p = self.params
        fast, slow = int(p["fast"]), int(p["slow"])
        if fast < 2 or slow < 3:
            raise ValueError("均线周期太小：快线至少 2，慢线至少 3")
        if fast >= slow:
            raise ValueError(f"快线周期({fast})必须小于慢线周期({slow})")
        if slow > 500:
            raise ValueError("慢线周期不要超过 500，否则预热期太长、可用数据不足")
        if str(p["ma_type"]).lower() not in ("sma", "ema"):
            raise ValueError("均线类型只能是 sma 或 ema")

    def warmup_bars(self) -> int:
        p = self.params
        return max(int(p["slow"]), int(p["trend_filter"]), 120) + 30

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        p = self.params
        fn = ind.ema if str(p["ma_type"]).lower() == "ema" else ind.sma
        c = df["close"]
        df["ma_fast"] = fn(c, int(p["fast"]))
        df["ma_slow"] = fn(c, int(p["slow"]))
        if int(p["trend_filter"]) > 0:
            df["ma_trend_f"] = ind.sma(c, int(p["trend_filter"]))
        return df

    def entry(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        gap = float(p["confirm_pct"]) / 100.0
        if gap > 0:
            # 用"确认幅度"替代纯金叉：快线需高出慢线一定比例，
            # 且前一日尚未达到该幅度，避免在缠绕区反复触发
            above = df["ma_fast"] >= df["ma_slow"] * (1 + gap)
            sig = safe(above & ~above.shift(1).fillna(False).astype(bool))
        else:
            sig = cross_above(df["ma_fast"], df["ma_slow"])
        if int(p["trend_filter"]) > 0:
            sig = safe(sig & (df["close"] > df["ma_trend_f"]))
        return sig

    def exit(self, df: pd.DataFrame) -> pd.Series:
        return cross_below(df["ma_fast"], df["ma_slow"])

    def reason(self, row: pd.Series, action: str) -> str:
        p = self.params
        t = str(p["ma_type"]).upper()
        f, s = int(p["fast"]), int(p["slow"])
        if action == "BUY":
            return (f"{t}{f} ({row['ma_fast']:.2f}) 上穿 {t}{s} ({row['ma_slow']:.2f})，"
                    f"高出 {(row['ma_fast'] / row['ma_slow'] - 1):.2%}")
        return (f"{t}{f} ({row['ma_fast']:.2f}) 下穿 {t}{s} ({row['ma_slow']:.2f})")


@register
class RsiReversal(Strategy):
    name = "rsi_reversal"
    label = "强势股回调低吸"
    description = (
        "只在长期趋势向上的股票里做（收盘价站上 MA120），等 RSI 跌进超卖区且当日收阳时买入，"
        "RSI 回到高位或跌破 MA60 时卖出。逆势低吸，胜率高但单笔盈亏比一般。"
    )
    defaults = {
        "rsi_buy": 32, "rsi_sell": 68,
        "trend_ma": 120, "stop_ma": 60,
        "max_dd": -0.30,   # 距离阶段高点回撤超过此值视为趋势已破，不接
    }
    param_meta = {
        "rsi_buy": {"label": "RSI 买入线", "min": 5, "max": 50, "step": 1,
                    "hint": "RSI 低于此值视为超卖。越低信号越少、越极端"},
        "rsi_sell": {"label": "RSI 卖出线", "min": 50, "max": 95, "step": 1},
        "trend_ma": {"label": "长期趋势均线", "min": 20, "max": 250, "step": 10,
                     "hint": "只在收盘价站上该均线的股票里做，避免下跌途中接刀"},
        "stop_ma": {"label": "止损均线", "min": 10, "max": 250, "step": 10},
        "max_dd": {"label": "最大回撤容忍", "min": -1, "max": 0, "step": 0.05,
                   "hint": "-0.3 表示距阶段高点已跌超 30% 的不接"},
    }

    def entry(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        uptrend = df["close"] > df[f"ma{p['trend_ma']}"]
        oversold = df["rsi14"] < p["rsi_buy"]
        # 要求当日收阳，避免在下跌途中连续接刀
        green = df["close"] > df["open"]
        not_broken = df["dd"] > p["max_dd"]
        return safe(uptrend & oversold & green & not_broken)

    def exit(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        overbought = cross_above(df["rsi14"], pd.Series(p["rsi_sell"], index=df.index))
        break_ma = df["close"] < df[f"ma{p['stop_ma']}"]
        return safe(overbought | break_ma)

    def score(self, df: pd.DataFrame) -> pd.Series:
        # RSI 越低越优先，同时偏好长期动量强的
        return (100 - df["rsi14"]).fillna(0) + df["mom120"].fillna(0) * 20

    def reason(self, row: pd.Series, action: str) -> str:
        if action == "BUY":
            return (f"长期趋势向上（MA120 {row['ma120']:.2f}），"
                    f"RSI 回落至 {row['rsi14']:.0f} 且当日收阳")
        return f"RSI 升至 {row['rsi14']:.0f} 或跌破 MA60 {row['ma60']:.2f}"


@register
class MomentumRotation(Strategy):
    name = "momentum_rotation"
    label = "动量轮动"
    description = (
        "横截面策略：每期在全市场按 60 日动量排名，买入排名靠前且站上 MA60 的股票，"
        "掉出排名区间或跌破 MA60 就换掉。适合搭配较低的调仓频率（周频/月频）。"
    )
    defaults = {
        "lookback": 60,
        "keep_rank": 0.10,   # 进入前 10% 才买
        "drop_rank": 0.30,   # 掉出前 30% 就卖
        "min_ma": 60,
    }
    param_meta = {
        "lookback": {"label": "动量回看(日)", "min": 10, "max": 250, "step": 10},
        "keep_rank": {"label": "买入排名分位", "min": 0.01, "max": 1, "step": 0.05},
        "drop_rank": {"label": "卖出排名分位", "min": 0.01, "max": 1, "step": 0.05},
        "min_ma": {"label": "趋势均线", "min": 10, "max": 250, "step": 10,
                   "hint": "站上才买、跌破就卖"},
    }

    def entry(self, df: pd.DataFrame) -> pd.Series:
        # 个股层面只做趋势过滤，真正的排名筛选交给引擎的 score + max_positions
        p = self.params
        above = df["close"] > df[f"ma{p['min_ma']}"]
        rising = df[f"ma{p['min_ma']}"] > df[f"ma{p['min_ma']}"].shift(10)
        pos_mom = df[f"mom{p['lookback']}"] > 0
        return safe(above & rising & pos_mom)

    def exit(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        return safe((df["close"] < df[f"ma{p['min_ma']}"]) |
                    (df[f"mom{p['lookback']}"] < 0))

    def score(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        # 动量除以波动率，偏好"走得稳的强势股"而非暴涨暴跌的
        mom = df[f"mom{p['lookback']}"]
        return (mom / df["vol20"].replace(0, np.nan)).fillna(-9.9)

    def reason(self, row: pd.Series, action: str) -> str:
        if action == "BUY":
            return (f"60日动量 {row['mom60']:+.1%}，年化波动 {row['vol20']:.1%}，"
                    f"站上 MA60 {row['ma60']:.2f}")
        return f"动量转负或跌破 MA60 {row['ma60']:.2f}"


@register
class RegimeMomentum(Strategy):
    name = "regime_momentum"
    label = "择时动量"
    description = (
        "在前面几个策略暴露的问题上重新设计的：①用沪深300 是否站上 200 日均线做大盘择时，"
        "空头市场一律不开新仓——这是最大的一个漏洞，选股再好也救不了满仓做多一个下跌市；"
        "②动量剔除最近 1 个月（学界的 12-1 动量做法），避开短期反转；"
        "③动量和波动率都设上限，不追半年已翻倍、也不碰年化波动 60% 以上的——"
        "无上限时买入标的的动量中位数高达 +103%，一半以上已翻倍，那是泡沫尾端；"
        "④只在自己的 MA60 上方买、跌破才走，让少数大赢家跑满。"
        "\n上限的依据是先验加「两段样本上回撤都下降」，不是收益——"
        "收益方向在样本内外相反，无法确认。详见 README 的样本外评估。"
    )
    defaults = {
        "index_ma": 200,      # 大盘择时均线（沪深300）
        "regime_confirm": 1,  # 择时状态翻转需连续确认的天数，1=不确认
        "lookback": 120,      # 动量回看窗口
        "skip_recent": 20,    # 动量剔除最近 N 日，规避短期反转
        "hold_ma": 60,        # 入场均线：站上才买
        "exit_ma": 0,         # 离场均线，0=与入场同一条。设更长的值可让赢家多跑
        "min_mom": 0.0,       # 动量下限，低于此值不买
        "max_mom": 1.0,       # 动量上限：半年已翻倍的不追
        "max_vol": 0.60,      # 年化波动率上限，滤掉投机性过强的
    }
    param_meta = {
        "index_ma": {"label": "大盘择时均线", "min": 0, "max": 400, "step": 10,
                     "hint": "沪深300 站上该均线才允许开新仓。0=关闭择时"},
        "regime_confirm": {"label": "择时确认天数", "min": 1, "max": 30, "step": 1,
                           "hint": "翻转需连续确认这么多天。⚠ 八年逐年实测无一致效果："
                                   "跑赢基准年数恒为 4/8，空仓天数仅降 2%，"
                                   "各设置间的差异呈非单调跳动（噪声特征）。默认关闭。"},
        "lookback": {"label": "动量回看(日)", "min": 20, "max": 500, "step": 10},
        "skip_recent": {"label": "剔除最近(日)", "min": 0, "max": 60, "step": 5,
                        "hint": "动量计算跳过最近 N 日，规避短期反转效应"},
        "hold_ma": {"label": "入场均线", "min": 10, "max": 250, "step": 10,
                    "hint": "收盘价站上该均线才买"},
        "exit_ma": {"label": "离场均线", "min": 0, "max": 250, "step": 10,
                    "hint": "0=与入场同一条。设更长即「快进慢出」。⚠ 八年实测：持有天数"
                            "和平均盈利单调变好（22→110日、+22%→+65%）、最差年份单调"
                            "改善（-33%→-24%），但收益无法确认——逐年与全程连续两种"
                            "评估给出的排名完全相反（MA120 逐年最好/全程最差）。"},
        "min_mom": {"label": "动量下限", "min": -1, "max": 5, "step": 0.05},
        "max_mom": {"label": "动量上限", "min": 0, "max": 20, "step": 0.25,
                    "hint": "1.0 表示回看期涨幅超过 100% 的不买。0=不限制。"
                            "极端动量往往是泡沫尾端，回撤极大"},
        "max_vol": {"label": "波动率上限", "min": 0, "max": 3, "step": 0.1,
                    "hint": "0.6 表示年化波动率超过 60% 的不买。0=不限制"},
    }

    def _exit_n(self) -> int:
        p = self.params
        return int(p["exit_ma"]) or int(p["hold_ma"])

    def warmup_bars(self) -> int:
        p = self.params
        return (int(p["lookback"]) + int(p["skip_recent"])
                + max(int(p["hold_ma"]), self._exit_n()) + 40)

    def market_regime(self, dates):
        n = int(self.params["index_ma"])
        if n <= 0:
            return None
        raw = index_above_ma(dates, "IDX000300", n)
        if raw is None:
            return None
        return hysteresis(raw, int(self.params["regime_confirm"]))

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        p = self.params
        c = df["close"]
        lb, sk = int(p["lookback"]), int(p["skip_recent"])
        # 12-1 式动量：跳过最近 sk 日，只看更早那段的涨幅
        df["mom_sk"] = c.shift(sk) / c.shift(sk + lb) - 1
        df["ma_hold"] = ind.sma(c, int(p["hold_ma"]))
        df["ma_exit"] = (df["ma_hold"] if not int(p["exit_ma"])
                         else ind.sma(c, int(p["exit_ma"])))
        return df

    def entry(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        sig = ((df["close"] > df["ma_hold"])
               & (df["ma_hold"] > df["ma_hold"].shift(20))
               & (df["mom_sk"] > float(p["min_mom"])))
        # 上限约束的依据是先验而非回测：动量效应在温和区间成立，
        # 半年翻倍以上的多是泡沫尾端，均值回归风险远大于延续性。
        # 实测未加上限时，买入标的的 120 日动量中位数高达 +103%，一半以上已翻倍。
        if float(p["max_mom"]) > 0:
            sig &= df["mom_sk"] <= float(p["max_mom"])
        if float(p["max_vol"]) > 0:
            sig &= df["vol20"] <= float(p["max_vol"])
        return safe(sig)

    def exit(self, df: pd.DataFrame) -> pd.Series:
        return safe(df["close"] < df["ma_exit"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        # 动量除以波动率：偏好走得稳的强势股，而非暴涨暴跌的
        return (df["mom_sk"] / df["vol20"].replace(0, np.nan)).fillna(-9.9)

    def reason(self, row: pd.Series, action: str) -> str:
        p = self.params
        if action == "BUY":
            return (f"大盘在 MA{p['index_ma']} 上方；剔除近 {p['skip_recent']} 日后的 "
                    f"{p['lookback']} 日动量 {row['mom_sk']:+.1%}，"
                    f"站上 MA{p['hold_ma']} ({row['ma_hold']:.2f})")
        return f"跌破 MA{self._exit_n()} ({row['ma_exit']:.2f})"


@register
class TurtleBreakout(Strategy):
    name = "turtle_breakout"
    label = "唐奇安通道突破"
    description = (
        "海龟法则的简化版：突破 N 日最高价买入，跌破 M 日最低价卖出。"
        "规则极简、参数少，不容易过拟合，适合当作衡量其他策略的基准线。"
    )
    defaults = {"entry_days": 20, "exit_days": 10}
    param_meta = {
        "entry_days": {"label": "入场通道(日)", "min": 5, "max": 250, "step": 5,
                       "hint": "突破过去 N 日最高价买入"},
        "exit_days": {"label": "离场通道(日)", "min": 3, "max": 120, "step": 5,
                      "hint": "跌破过去 M 日最低价卖出。通常取入场通道的一半"},
    }

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        p = self.params
        df["dc_up"] = ind.rolling_high(df["high"], p["entry_days"]).shift(1)
        df["dc_dn"] = ind.rolling_low(df["low"], p["exit_days"]).shift(1)
        return df

    def entry(self, df: pd.DataFrame) -> pd.Series:
        return safe(df["close"] > df["dc_up"])

    def exit(self, df: pd.DataFrame) -> pd.Series:
        # 同 MaTrend：跟踪止损归引擎管，这里只判通道下轨
        return safe(df["close"] < df["dc_dn"])

    def reason(self, row: pd.Series, action: str) -> str:
        if action == "BUY":
            return f"突破 {self.params['entry_days']} 日通道上轨 {row['dc_up']:.2f}"
        return f"跌破 {self.params['exit_days']} 日通道下轨 {row['dc_dn']:.2f}"
