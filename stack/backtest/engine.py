"""回测引擎：逐日事件驱动，按 A 股实际规则撮合。

刻意建模的几件事，缺一个回测结果就会系统性偏乐观：

1. 信号日收盘产生信号，**次日开盘**成交。当日信号当日成交是最常见的未来函数。
2. T+1：因为买入发生在次日开盘，卖出最早也在再下一日，天然满足。
3. 涨跌停：次日开盘一字涨停就买不进、跌停就卖不出，按板块区分 10%/20%，ST 减半。
4. 停牌：当日无 K 线数据即视为不可交易，持仓继续按最后价格计价。
5. 交易成本：佣金（含最低 5 元）、卖出印花税、过户费，买卖分别计。
6. 100 股整手，资金不足买一手就跳过。

这些约束会让漂亮的曲线变难看——那才是接近真实的样子。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..config import LOT_SIZE, buy_cost, price_limit, sell_cost
from ..data import store
from ..strategies.base import Strategy

EPS = 1e-6


@dataclass
class BacktestConfig:
    initial_cash: float = 200_000.0
    max_positions: int = 5           # 同时最多持有几只
    position_pct: float = 0.0        # >0 则每仓固定占总资产比例，否则按 1/max_positions
    stop_loss: float = 0.0           # 相对成本价的硬止损，如 0.08 表示亏 8% 无条件卖
    take_profit: float = 0.0         # 硬止盈，0 为不启用
    trail_stop_atr: float = 0.0      # 跟踪止损：自持仓期间最高价回落 N×ATR 就卖，0 为不启用
    max_hold_days: int = 0           # 最长持有天数，0 为不限
    min_hold_days: int = 1           # 最短持有天数，防止信号抖动导致次日就卖
    warmup_days: int = 400           # 指标预热窗口（自然日）


@dataclass
class Trade:
    code: str
    name: str
    open_date: str
    close_date: str
    shares: int
    open_price: float
    close_price: float
    pnl: float                # 已扣费净盈亏
    pnl_pct: float
    hold_days: int
    open_reason: str
    close_reason: str


@dataclass
class BacktestResult:
    strategy: str
    params: dict
    config: dict
    start: str
    end: str
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)   # 因涨跌停/停牌被拦下的次数
    analytics: dict = field(default_factory=dict)  # 回撤曲线/月度收益等，给界面用

    def to_json(self) -> dict:
        eq = self.equity
        return {
            "strategy": self.strategy,
            "params": self.params,
            "config": self.config,
            "start": self.start,
            "end": self.end,
            "metrics": self.metrics,
            "skipped": self.skipped,
            "equity": {
                "dates": eq["date"].tolist() if not eq.empty else [],
                "value": [round(v, 2) for v in eq["equity"]] if not eq.empty else [],
                "benchmark": ([round(v, 2) for v in eq["benchmark"]]
                              if not eq.empty and "benchmark" in eq else []),
            },
            "analytics": self.analytics,
            "trades": [asdict(t) for t in self.trades],
        }


# ------------------------------------------------------------------ 数据准备
class Panel:
    """把所有股票的信号展平成 (股票数 × 交易日数) 的二维 numpy 数组。

    为什么不直接用 {code: DataFrame} + .loc 查找：回测内层循环是
    「每个交易日 × 每只股票」，5 年 × 900 只就是 120 万次查找。pandas 每次 .loc
    都要构造一个 Series，开销会完全主导运行时间。展平成 numpy 后是纯整数索引，
    候选筛选还能整列向量化，实测快一个数量级以上。
    """

    __slots__ = ("codes", "idx", "dates", "open", "high", "close",
                 "atr", "entry", "exit", "score", "valid")

    def __init__(self, codes, dates, arrays):
        self.codes = codes
        self.idx = {c: i for i, c in enumerate(codes)}
        self.dates = dates
        for k, v in arrays.items():
            setattr(self, k, v)

    def __len__(self):
        return len(self.codes)


def _prepare_panel(strategy: Strategy, codes: list[str], start: str, end: str,
                   warmup_days: int, dates: list[str]) -> Panel | None:
    # 预热窗口取"配置值"和"策略自称所需"的较大者。参数可调的策略（如双均线把慢线
    # 设到 250）需要的历史比默认值长得多，喂不够会静默跑不出信号。
    # 250 个交易日约合 365 个自然日，故按 1.5 倍换算并留出停牌余量。
    need = int(strategy.warmup_bars() * 1.5) + 30
    load_start = (pd.Timestamp(start)
                  - timedelta(days=max(warmup_days, need))).strftime("%Y-%m-%d")
    raw = store.load_daily(codes=codes, start=load_start, end=end)
    if raw.empty:
        return None

    date_pos = {d: i for i, d in enumerate(dates)}
    n_d = len(dates)
    keep_codes: list[str] = []
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("open", "high", "close", "atr", "score")}
    flags: dict[str, list[np.ndarray]] = {k: [] for k in ("entry", "exit", "valid")}
    # 多因子打分声明的原始列，逐股票收集，最后由引擎做横截面归一
    score_cols = [n for n, _ in getattr(strategy, "score_fields", []) or []]
    extra_lists: dict[str, list[np.ndarray]] = {n: [] for n in score_cols}

    for code, g in raw.groupby("code", sort=False):
        g = store.usable_history(g.sort_values("date"))
        if len(g) < 60:                       # 数据太短，指标算不出来
            continue
        try:
            d = strategy.prepare(g)
            entry = np.asarray(strategy.entry(d), dtype=bool)
            exit_ = np.asarray(strategy.exit(d), dtype=bool)
            score = np.asarray(strategy.score(d), dtype=np.float32)
        except Exception:
            continue

        # 只保留落在回测区间内的行，映射到全局日期轴
        ds = d["date"].dt.strftime("%Y-%m-%d").to_numpy()
        pos = np.array([date_pos.get(x, -1) for x in ds])
        sel = pos >= 0
        if not sel.any():
            continue
        pos = pos[sel]

        row = {k: np.full(n_d, np.nan, dtype=np.float32)
               for k in ("open", "high", "close", "atr", "score")}
        row["open"][pos] = d["open"].to_numpy()[sel]
        row["high"][pos] = d["high"].to_numpy()[sel]
        row["close"][pos] = d["close"].to_numpy()[sel]
        row["atr"][pos] = d["atr14"].to_numpy()[sel]
        row["score"][pos] = score[sel]

        for name in score_cols:
            arr = np.full(n_d, np.nan, dtype=np.float32)
            if name in d.columns:
                arr[pos] = pd.to_numeric(d[name], errors="coerce").to_numpy()[sel]
            extra_lists[name].append(arr)

        e = np.zeros(n_d, dtype=bool); e[pos] = entry[sel]
        x = np.zeros(n_d, dtype=bool); x[pos] = exit_[sel]
        v = np.zeros(n_d, dtype=bool); v[pos] = True

        keep_codes.append(code)
        for k in cols:
            cols[k].append(row[k])
        flags["entry"].append(e)
        flags["exit"].append(x)
        flags["valid"].append(v)

    if not keep_codes:
        return None
    arrays = {k: np.vstack(v) for k, v in cols.items()}
    arrays.update({k: np.vstack(v) for k, v in flags.items()})
    extra = {k: np.vstack(v) for k, v in extra_lists.items() if v}

    fields = getattr(strategy, "score_fields", None)
    if fields:
        # 每个分量按日期逐列转成横截面百分位排名，再加权求和。
        # 这是唯一能正确归一的地方——策略里只看得到单只股票的时序，
        # 在那里做 rank 排的是时间维度且会用到未来数据。
        total = np.zeros_like(arrays["score"], dtype=np.float32)
        wsum = 0.0
        for name, w in fields:
            m = extra.get(name)
            if m is None:
                continue
            total += w * _cross_rank(m, arrays["valid"])
            wsum += w
        arrays["score"] = (total / wsum if wsum else total).astype(np.float32)
        arrays["score"][~arrays["valid"]] = -9.9

    return Panel(keep_codes, dates, arrays)


def _cross_rank(mat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把 (股票 × 日期) 矩阵按日期逐列转成 [0,1] 的横截面百分位排名。

    无效值（NaN 或当日不可交易）不参与排名，统一给 0。
    """
    s = mat.astype(np.float64).copy()
    s[~valid] = np.nan
    bad = np.isnan(s)
    filled = np.where(bad, -np.inf, s)
    order = np.argsort(np.argsort(filled, axis=0), axis=0).astype(np.float64)
    n_valid = (~bad).sum(axis=0)
    # argsort 把 -inf 排在最前，有效值的名次要减掉无效值个数
    order -= (s.shape[0] - n_valid)[None, :]
    pct = np.where(n_valid[None, :] > 1,
                   order / np.maximum(n_valid[None, :] - 1, 1), 0.5)
    return np.where(bad, 0.0, pct).astype(np.float32)


def _benchmark(start: str, end: str, symbol: str = "IDX000300") -> pd.Series | None:
    """沪深300 作为基准。本地库里没有指数就返回 None，不影响回测。"""
    df = store.load_daily(codes=[symbol], start=start, end=end)
    if df.empty:
        return None
    df = df.sort_values("date")
    return pd.Series(df["close"].to_numpy(),
                     index=df["date"].dt.strftime("%Y-%m-%d"))


# ------------------------------------------------------------------ 主循环
class _Position:
    __slots__ = ("i", "code", "name", "shares", "cost", "open_date",
                 "open_reason", "peak", "hold_days")

    def __init__(self, i, code, name, shares, cost, open_date, open_reason):
        self.i = i                    # 在 Panel 里的行号
        self.code = code
        self.name = name
        self.shares = shares
        self.cost = cost              # 含费成本价
        self.open_date = open_date
        self.open_reason = open_reason
        self.peak = cost
        self.hold_days = 0


def run(strategy: Strategy, codes: list[str], start: str, end: str,
        cfg: BacktestConfig | None = None,
        names: dict[str, str] | None = None) -> BacktestResult:
    cfg = cfg or BacktestConfig()
    names = names or {}

    result = BacktestResult(
        strategy=strategy.name, params=strategy.params,
        config=asdict(cfg), start=start, end=end,
    )
    dates = store.trading_days(start=start, end=end)
    if len(dates) < 2:
        result.metrics = {"error": "交易日不足，无法回测"}
        return result

    P = _prepare_panel(strategy, codes, start, end, cfg.warmup_days, dates)
    if P is None:
        result.metrics = {"error": "所选区间内没有可用数据，请先同步行情"}
        return result

    # 涨跌停幅度只和代码、是否 ST 有关，预先算好，免得在内层循环里反复判断
    limits = np.array([price_limit(c, names.get(c, "")) for c in P.codes],
                      dtype=np.float64)

    # 大盘择时：只压制开仓，绝不阻止平仓——否则大盘转空时反而被锁在里面出不来
    regime = strategy.market_regime(dates)
    regime_arr = (None if regime is None
                  else regime.reindex(dates).fillna(False).to_numpy(dtype=bool))

    cash = cfg.initial_cash
    positions: dict[int, _Position] = {}
    equity_rows: list[dict] = []
    skipped = {"涨停无法买入": 0, "跌停无法卖出": 0, "停牌": 0, "资金不足": 0}
    regime_off_days = 0
    pct_alloc = cfg.position_pct if cfg.position_pct > 0 else 1.0 / cfg.max_positions

    bench = _benchmark(start, end)
    bench_base = None

    for j in range(1, len(dates)):
        today = dates[j]
        op_t, cl_p = P.open[:, j], P.close[:, j - 1]
        valid_t, valid_p = P.valid[:, j], P.valid[:, j - 1]

        # ---------------------------------------------------- 1. 卖出
        for i in list(positions):
            pos = positions[i]
            pos.hold_days += 1
            if not (valid_t[i] and valid_p[i]):
                skipped["停牌"] += 1
                continue

            prev_close = float(cl_p[i])
            reason = None
            if pos.hold_days >= cfg.min_hold_days:
                if P.exit[i, j - 1]:
                    # 引擎持有的是展平后的 numpy 数组，拿不到 pandas 行，因此这里
                    # 只能给出概括性理由。每日信号页走的是 strategy.reason()，
                    # 有完整的指标数值——那才是你实际据以决策的地方。
                    reason = f"{strategy.label}：触发离场信号（前收 {prev_close:.2f}）"
                elif cfg.stop_loss > 0 and prev_close <= pos.cost * (1 - cfg.stop_loss):
                    reason = f"触发止损 -{cfg.stop_loss:.0%}（成本 {pos.cost:.2f}）"
                elif cfg.take_profit > 0 and prev_close >= pos.cost * (1 + cfg.take_profit):
                    reason = f"触发止盈 +{cfg.take_profit:.0%}（成本 {pos.cost:.2f}）"
                elif cfg.max_hold_days > 0 and pos.hold_days >= cfg.max_hold_days:
                    reason = f"持有满 {cfg.max_hold_days} 日到期"
                elif cfg.trail_stop_atr > 0:
                    # 跟踪止损必须在这里判：只有引擎知道持仓期内的最高价
                    atr = float(P.atr[i, j - 1])
                    if atr == atr and pos.peak - cfg.trail_stop_atr * atr >= prev_close:
                        reason = (f"跟踪止损：自持仓最高 {pos.peak:.2f} 回落超过 "
                                  f"{cfg.trail_stop_atr}×ATR({atr:.2f})")
            if reason is None:
                continue

            # 开盘跌停则卖不出，顺延到之后的交易日
            px = float(op_t[i])
            if px <= 0:
                # 前复权价可能为负（累计分红超过当年股价），此时成交价无意义。
                # 买入侧已有同样的守卫，卖出侧不能漏——负价格会算出负的卖出所得。
                skipped["停牌"] += 1
                continue
            if px <= prev_close * (1 - limits[i]) + EPS:
                skipped["跌停无法卖出"] += 1
                continue

            gross = px * pos.shares
            proceeds = gross - sell_cost(gross)
            cash += proceeds
            pnl = proceeds - pos.cost * pos.shares
            result.trades.append(Trade(
                code=pos.code, name=pos.name, open_date=pos.open_date, close_date=today,
                shares=pos.shares, open_price=round(pos.cost, 3), close_price=round(px, 3),
                pnl=round(pnl, 2), pnl_pct=round(pnl / (pos.cost * pos.shares), 4),
                hold_days=pos.hold_days,
                open_reason=pos.open_reason, close_reason=reason,
            ))
            del positions[i]

        # ---------------------------------------------------- 2. 买入
        # 用第 j-1 日的择时状态决定第 j 日能否开仓，与信号口径一致
        regime_on = regime_arr is None or bool(regime_arr[j - 1])
        if not regime_on:
            regime_off_days += 1
        slots = (cfg.max_positions - len(positions)) if regime_on else 0
        if slots > 0:
            # 整列向量化筛候选：昨日有买入信号、昨天和今天都能交易、且当前未持有
            mask = P.entry[:, j - 1] & valid_t & valid_p
            if positions:
                mask[list(positions)] = False
            cand = np.flatnonzero(mask)
            if cand.size:
                sc = P.score[cand, j - 1]
                sc = np.where(np.isfinite(sc), sc, -np.inf)
                cand = cand[np.argsort(-sc, kind="stable")]

                # 每仓资金按当前总资产而非初始资金，让盈利复投
                mv_now = sum(p.shares * float(P.close[p.i, j])
                             for p in positions.values() if valid_t[p.i])
                budget = (cash + mv_now) * pct_alloc

                for ci in cand:
                    if slots <= 0:
                        break
                    i = int(ci)
                    px, prev_close = float(op_t[i]), float(cl_p[i])
                    if px <= 0:
                        continue
                    if px >= prev_close * (1 + limits[i]) - EPS:   # 一字涨停买不到
                        skipped["涨停无法买入"] += 1
                        continue

                    shares = int(budget / px // LOT_SIZE) * LOT_SIZE
                    while shares > 0 and px * shares + buy_cost(px * shares) > cash:
                        shares -= LOT_SIZE
                    if shares <= 0:
                        skipped["资金不足"] += 1
                        continue

                    gross = px * shares
                    fee = buy_cost(gross)
                    cash -= gross + fee
                    code = P.codes[i]
                    positions[i] = _Position(
                        i=i, code=code, name=names.get(code, code), shares=shares,
                        cost=(gross + fee) / shares, open_date=today,
                        open_reason=(f"{strategy.label}：触发入场信号"
                                     f"（前收 {prev_close:.2f}，评分 "
                                     f"{float(P.score[i, j - 1]):.3f}）"),
                    )
                    slots -= 1

        # ---------------------------------------------------- 3. 盯市
        mv = 0.0
        for pos in positions.values():
            if valid_t[pos.i]:
                c = float(P.close[pos.i, j])
                pos.peak = max(pos.peak, c)
                mv += pos.shares * c
            else:
                mv += pos.shares * pos.cost      # 停牌按成本价计
        row = {"date": today, "equity": cash + mv,
               "cash": cash, "positions": len(positions)}
        if bench is not None and today in bench.index:
            if bench_base is None:
                bench_base = float(bench.loc[today])
            row["benchmark"] = cfg.initial_cash * float(bench.loc[today]) / bench_base
        equity_rows.append(row)

    result.equity = pd.DataFrame(equity_rows)
    result.skipped = {k: v for k, v in skipped.items() if v}
    if regime_arr is not None:
        result.skipped["大盘择时空仓天数"] = regime_off_days
    from .metrics import analytics, compute
    result.metrics = compute(result.equity, result.trades, cfg.initial_cash)
    result.analytics = analytics(result.equity, result.trades, cfg.initial_cash)
    return result
