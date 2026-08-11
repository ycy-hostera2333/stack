"""自检：把开发期验证过的不变量固化下来，改代码后一条命令重跑。

    python -m stack.cli selftest

这些检查针对的是「静默错误」——不会抛异常、但会让回测结果系统性偏乐观的那类问题。
它们比单元测试更重要，因为一个算错的回测不会报错，只会让你亏钱。

不依赖 pytest，保持零额外依赖。
"""
from __future__ import annotations

import time
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd

from . import indicators as ind
from .backtest import engine
from .config import buy_cost, price_limit, sell_cost
from .data import store, universe
from .strategies import all_strategies, get_strategy
from .strategies.base import Strategy, safe

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """装饰器：把函数注册成一项检查，异常/断言失败都记为不通过。"""
    def deco(fn):
        def run():
            t0 = time.time()
            try:
                msg = fn() or "通过"
                _RESULTS.append((name, True, f"{msg}  ({time.time()-t0:.1f}s)"))
            except AssertionError as e:
                _RESULTS.append((name, False, f"断言失败：{e}"))
            except Exception as e:
                _RESULTS.append((name, False,
                                 f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
        run.__name__ = fn.__name__
        _CHECKS.append(run)
        return run
    return deco


_CHECKS: list = []


# ------------------------------------------------------------------ 指标
@check("指标：MA/RSI/ATR 在构造数据上取值正确")
def _t_indicators():
    s = pd.Series([1.0] * 10)
    assert ind.sma(s, 5).iloc[-1] == 1.0, "常数序列的均线应等于该常数"
    assert pd.isna(ind.sma(s, 5).iloc[3]), "窗口不足时必须是 NaN，不能提前给值"

    up = pd.Series(np.arange(1, 41, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] > 99, f"单调上涨的 RSI 应接近 100，实际 {ind.rsi(up,14).iloc[-1]}"
    down = pd.Series(np.arange(40, 0, -1, dtype=float))
    assert ind.rsi(down, 14).iloc[-1] < 1, "单调下跌的 RSI 应接近 0"

    n = 30
    h = pd.Series([11.0] * n); l = pd.Series([9.0] * n); c = pd.Series([10.0] * n)
    assert abs(ind.atr(h, l, c, 14).iloc[-1] - 2.0) < 1e-6, "恒定 2 元振幅的 ATR 应为 2"

    # 所有指标都不能用到未来数据：截断尾部后，前面的取值必须完全不变
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .02, 400))))
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * .99,
                       "close": px, "volume": 1e6})
    full = ind.add_common(df)
    part = ind.add_common(df.iloc[:300])
    for col in ("ma20", "ma60", "rsi14", "atr14", "mom60", "vol20"):
        a = full[col].iloc[:300].to_numpy()
        b = part[col].to_numpy()
        m = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[m], b[m]), f"{col} 依赖了未来数据：截断后取值改变"
    return "含未来函数检查"


# ------------------------------------------------------------------ 费用
@check("费用：佣金最低 5 元、印花税单边收取")
def _t_cost():
    assert abs(buy_cost(1000) - (5 + 1000 * 1e-5)) < 1e-9, "小额买入应触发 5 元最低佣金"
    big = 1_000_000
    assert abs(buy_cost(big) - (big * 2.5e-4 + big * 1e-5)) < 1e-6
    # 卖出比买入多一道印花税
    assert abs((sell_cost(big) - buy_cost(big)) - big * 5e-4) < 1e-6, "印花税应只在卖出时收"
    assert price_limit("600519") == 0.10 and price_limit("300750") == 0.20
    assert price_limit("600519", "ST某某") == 0.05, "主板 ST 涨跌停应为 5%"
    assert price_limit("300750", "ST某某") == 0.20, "创业板 ST 仍为 20%"
    return "含 ST/板块涨跌停"


# ------------------------------------------------------------------ 策略
@check("策略：全部可实例化，参数校验能挡住非法组合")
def _t_strategies():
    infos = all_strategies()
    assert len(infos) >= 5, f"注册的策略太少：{len(infos)}"
    for info in infos:
        st = get_strategy(info["name"])
        assert st.warmup_bars() > 0
        assert isinstance(st.info()["defaults"], dict)

    # 未知参数必须报错，不能被静默忽略
    try:
        get_strategy("ma_cross", 不存在的参数=1)
        raise AssertionError("未知参数没有被拦截")
    except ValueError:
        pass
    # 快线 >= 慢线必须报错
    for kw in ({"fast": 30, "slow": 10}, {"fast": 20, "slow": 20}):
        try:
            get_strategy("ma_cross", **kw)
            raise AssertionError(f"非法组合 {kw} 没有被拦截")
        except ValueError:
            pass
    # 预热窗口必须跟着参数走，否则长周期会静默失效
    a = get_strategy("ma_cross", fast=5, slow=20).warmup_bars()
    b = get_strategy("ma_cross", fast=60, slow=250).warmup_bars()
    assert b > a, f"慢线变长后预热窗口没有增加（{a} -> {b}）"
    return f"{len(infos)} 个策略"


# ------------------------------------------------------------------ 回测引擎
def _sample(n=250, start="2022-01-01"):
    """取一小撮真实数据用于引擎检查；数据不足时返回 None 让检查跳过。"""
    uni = universe.build(as_of=start)
    if uni.empty or len(uni) < 40:
        return None
    codes = uni["code"].tolist()[:n]
    return codes, dict(zip(uni["code"], uni["name"]))


@check("引擎：会计恒等式（期末净值 = 本金 + 已实现 + 未实现）")
def _t_accounting():
    s = _sample()
    if s is None:
        return "跳过：本地数据不足"
    codes, names = s
    cfg = engine.BacktestConfig(initial_cash=200_000, max_positions=5,
                                trail_stop_atr=2.0)
    r = engine.run(get_strategy("turtle_breakout"), codes,
                   "2022-01-01", "2024-12-31", cfg, names)
    assert "error" not in r.metrics, r.metrics.get("error")
    realized = sum(t.pnl for t in r.trades)
    delta = float(r.equity["equity"].iloc[-1]) - cfg.initial_cash
    unreal = delta - realized
    assert abs((realized + unreal) - delta) < 1e-6, "现金流对不上"
    eq = r.equity["equity"].to_numpy()
    assert np.isfinite(eq).all(), "净值序列出现 NaN/Inf"
    assert (eq >= 0).all(), "净值出现负数"
    return f"{len(r.trades)} 笔交易，误差 < 1e-6"


@check("引擎：T+1（卖出日必须晚于买入日，持有 >= 1 日）")
def _t_t1():
    s = _sample()
    if s is None:
        return "跳过：本地数据不足"
    codes, names = s
    r = engine.run(get_strategy("ma_trend"), codes, "2022-01-01", "2024-12-31",
                   engine.BacktestConfig(max_positions=5), names)
    bad = [t for t in r.trades if t.close_date <= t.open_date or t.hold_days < 1]
    assert not bad, f"{len(bad)} 笔交易违反 T+1，例如 {bad[0].code} {bad[0].open_date}"
    return f"{len(r.trades)} 笔全部合规"


@check("引擎：无未来函数（截断数据后，截断点之前的交易完全不变）")
def _t_lookahead():
    s = _sample()
    if s is None:
        return "跳过：本地数据不足"
    codes, names = s
    st, cfg = get_strategy("turtle_breakout"), engine.BacktestConfig(max_positions=5)
    full = engine.run(st, codes, "2022-01-01", "2024-12-31", cfg, names)
    cut = "2023-12-29"
    trunc = engine.run(st, codes, "2022-01-01", cut, cfg, names)
    before = [t for t in full.trades if t.close_date <= cut]
    assert len(before) == len(trunc.trades), (
        f"截断后交易数变了：{len(before)} vs {len(trunc.trades)}——存在未来信息泄漏")
    for a, b in zip(before, trunc.trades):
        assert (a.code == b.code and a.open_date == b.open_date
                and a.close_date == b.close_date and abs(a.pnl - b.pnl) < 0.01), (
            f"截断后交易变了：{a.code} {a.open_date}")
    return f"{len(before)} 笔交易逐笔一致"


@check("引擎：成本模型（买入持有的回测结果 ≈ 标的实际等权涨跌幅）")
def _t_cost_model():
    s = _sample(n=200)
    if s is None:
        return "跳过：本地数据不足"
    codes, names = s

    class BuyHold(Strategy):
        name, label, defaults = "_bh", "买入持有", {}

        def entry(self, df):
            return safe(df["ma60"].notna())

        def exit(self, df):
            return safe(pd.Series(False, index=df.index))

        def score(self, df):
            return pd.Series(0.0, index=df.index)

    START, END = "2022-01-01", "2024-12-31"
    cfg = engine.BacktestConfig(initial_cash=20_000_000, max_positions=150)
    r = engine.run(BuyHold(), codes, START, END, cfg, names)
    assert "error" not in r.metrics, r.metrics.get("error")

    raw = store.load_daily(codes=codes, start=START, end=END)
    rets = [g.sort_values("date")["close"].iloc[-1] / g.sort_values("date")["close"].iloc[0] - 1
            for _, g in raw.groupby("code") if len(g) > 200]
    actual = float(np.mean(rets))
    got = r.metrics["total_return"]
    # 差异只应来自手续费、次日开盘买入和买不满的仓位，超过 8 个百分点就说明成本模型有系统性错误
    assert abs(got - actual) < 0.08, (
        f"买入持有 {got:+.2%} 与标的等权 {actual:+.2%} 相差过大，成本模型可能有误")
    return f"回测 {got:+.2%} vs 实际等权 {actual:+.2%}"


@check("引擎：涨跌停与停牌确实拦下了成交")
def _t_limits():
    s = _sample()
    if s is None:
        return "跳过：本地数据不足"
    codes, names = s
    r = engine.run(get_strategy("ma_trend"), codes, "2022-01-01", "2024-12-31",
                   engine.BacktestConfig(max_positions=5), names)
    keys = set(r.skipped)
    assert keys & {"涨停无法买入", "跌停无法卖出", "停牌"}, (
        f"三年回测里一次涨跌停/停牌都没拦下，规则可能没生效：{r.skipped}")
    return "、".join(f"{k} {v}" for k, v in r.skipped.items())


# ------------------------------------------------------------------ 数据
@check("模拟盘：与回测引擎逐笔等价（同一股票池、同一区间）")
def _t_paper_equiv():
    from . import paper
    from .data import universe as uni_mod

    days = store.trading_days()
    if len(days) < 40:
        return "跳过：交易日不足"
    end = days[-1]
    start = days[-31]                      # 最近约 30 个交易日
    eng_start = days[-32]                  # 引擎提前一日，使首个可交易日对齐

    fixed = uni_mod.build(as_of=start)
    if fixed.empty or len(fixed) < 50:
        return "跳过：股票池不足"
    codes = fixed["code"].tolist()[:150]
    names = dict(zip(fixed["code"], fixed["name"]))
    fdf = fixed[fixed["code"].isin(codes)]

    st = get_strategy("regime_momentum")
    r = engine.run(st, codes, eng_start, end,
                   engine.BacktestConfig(initial_cash=200_000, max_positions=5), names)

    # 固定股票池，排除「模拟盘每日重建池子」这一设计差异
    orig = uni_mod.build
    saved = {k: paper._meta(k) for k in ("strategy", "params", "cash",
                                         "initial_cash", "max_positions",
                                         "top", "last_date", "created_at")}
    uni_mod.build = lambda flt=None, as_of=None: fdf
    try:
        paper.reset("regime_momentum", {}, 200_000, 5, 150)
        for d in store.trading_days(start=start, end=end):
            paper.advance(as_of=d, verbose=False)
        with store.connect() as c:
            pt = pd.read_sql("SELECT code,open_date,close_date,shares,pnl "
                             "FROM paper_trade", c)
    finally:
        uni_mod.build = orig

    et = pd.DataFrame([{"code": t.code, "open_date": t.open_date,
                        "close_date": t.close_date, "shares": t.shares,
                        "pnl": round(t.pnl, 2)} for t in r.trades])
    if et.empty and pt.empty:
        return "区间内无交易，无法比对"
    assert len(et) == len(pt), f"成交笔数不同：引擎 {len(et)}，模拟盘 {len(pt)}"
    et = et.sort_values(["open_date", "code"]).reset_index(drop=True)
    pt = pt.sort_values(["open_date", "code"]).reset_index(drop=True)
    for col in ("code", "open_date", "close_date", "shares"):
        assert (et[col].values == pt[col].values).all(), f"{col} 不一致"
    assert (abs(et["pnl"].values - pt["pnl"].values) < 0.05).all(), "盈亏不一致"

    # 还原调用方原本的模拟盘配置，避免自检把用户的账户冲掉
    if saved.get("strategy"):
        import json as _json
        paper.reset(saved["strategy"], _json.loads(saved["params"] or "{}"),
                    float(saved["initial_cash"]), int(saved["max_positions"]),
                    int(saved["top"]))
    return f"{len(et)} 笔逐笔一致"


@check("数据：OHLC 逻辑自洽、主键无重复、无未来日期")
def _t_data():
    cov = store.coverage()
    if not cov["bars"]:
        return "跳过：本地还没有数据"
    with store.connect() as c:
        dup = c.execute("SELECT COUNT(*) FROM (SELECT code,date,COUNT(*) n "
                        "FROM daily GROUP BY code,date HAVING n>1)").fetchone()[0]
        bad = c.execute("SELECT COUNT(*) FROM daily WHERE low>high OR low>open "
                        "OR low>close OR high<open OR high<close").fetchone()[0]
    assert dup == 0, f"{dup} 组重复主键"
    assert bad == 0, f"{bad} 根 K 线的 OHLC 关系不成立"
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    assert cov["last_date"] <= today, f"出现未来日期 {cov['last_date']}"
    return f"{cov['bars']:,} 根 K 线，{cov['codes_with_data']} 只"


@check("数据：成交额列完整（缺失会让股票池静默塌陷）")
def _t_amount():
    cov = store.coverage()
    if not cov["bars"]:
        return "跳过：本地还没有数据"
    with store.connect() as c:
        rows = c.execute("""
            SELECT substr(date,1,7) ym, COUNT(*),
                   SUM(CASE WHEN amount IS NULL OR amount<=0 THEN 1 ELSE 0 END)
            FROM daily WHERE code NOT LIKE 'IDX%'
            GROUP BY ym ORDER BY ym""").fetchall()
    bad = [(ym, n, m) for ym, n, m in rows if n > 500 and m / n > 0.30]
    total_missing = sum(m for _, _, m in rows) / max(sum(n for _, n, _ in rows), 1)

    # 曾经踩过：腾讯源只返回 6 个字段（无成交额），而它排在 fallback 首位，
    # 结果 2024-07 之后 99% 的行 amount 为 NULL。universe.build 用 20 日均成交额
    # 筛流动性，于是股票池从两千多只塌成 17 只，2025 年之后的回测全部返回空——
    # 全程不报任何错。这一项就是为了让这种事下次立刻暴露。
    assert not bad, (
        f"{len(bad)} 个月份的成交额缺失超过 30%，最早 {bad[0][0]}"
        f"（{bad[0][2]}/{bad[0][1]}）。股票池会因此塌陷，回测结果不可信。")

    # 顺带验证量纲：amount ≈ volume(手) × 100 × 均价
    with store.connect() as c:
        r = c.execute("""
            SELECT AVG(amount/(volume*close)) FROM daily
            WHERE code NOT LIKE 'IDX%' AND volume>0 AND close>0
              AND amount>0 AND date>=date('now','-60 day')""").fetchone()[0]
    if r is not None:
        assert 50 < r < 200, f"amount/(volume×close) = {r:.1f}，量纲异常（应≈100）"
    return f"整体缺失 {total_missing:.1%}，量纲比值 {r:.0f}" if r else "通过"


@check("数据：负价历史被正确截断（前复权价可为负，不能喂给指标）")
def _t_nonpositive():
    with store.connect() as c:
        neg = c.execute("SELECT COUNT(*) FROM daily WHERE close<=0").fetchone()[0]
        codes = [r[0] for r in c.execute(
            "SELECT DISTINCT code FROM daily WHERE close<=0 LIMIT 20")]
    if not codes:
        return "本地库中没有负价 K 线"

    kept = dropped = 0
    for code in codes:
        g = store.load_daily([code]).sort_values("date")
        u = store.usable_history(g)
        assert (u["close"] > 0).all(), f"{code} 截断后仍有非正收盘价"
        assert (u["low"] > 0).all(), f"{code} 截断后仍有非正最低价"
        if len(u):
            assert u["date"].iloc[-1] == g["date"].iloc[-1], f"{code} 误删了最新数据"
        kept += len(u); dropped += len(g) - len(u)

    # 指标必须建立在截断后的数据上
    g = store.load_daily([codes[0]]).sort_values("date")
    d = ind.add_common(store.usable_history(g))
    for col in ("ma20", "ma60", "rsi14"):
        v = d[col].dropna()
        assert (v > 0).all() if col.startswith("ma") else True, f"{col} 出现非正值"
    return (f"{neg:,} 根负价 K 线分布在 {len(codes)} 只股票，"
            f"抽查截断 {dropped:,} 根、保留 {kept:,} 根")


@check("数据：盘中残缺 K 线防护（未收盘时信号必须回退到上一交易日）")
def _t_partial_bar():
    from . import signals as sig
    from .data import source
    days = store.trading_days()
    if len(days) < 2:
        return "跳过：交易日不足"
    as_of, skipped = sig._latest_usable_date(allow_partial=False)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    if days[-1] == today and not source.market_closed_today():
        assert skipped and as_of == days[-2], (
            f"当前未收盘却仍在用当日 K 线：as_of={as_of}")
        return f"已回退到 {as_of}（当前未收盘）"
    assert as_of == days[-1] and not skipped
    return f"使用 {as_of}（已收盘或非交易日）"


# ------------------------------------------------------------------ 入口
def run_all(verbose: bool = True) -> bool:
    _RESULTS.clear()
    store.init_db()
    t0 = time.time()
    for fn in _CHECKS:
        fn()

    ok = sum(1 for _, p, _ in _RESULTS if p)
    if verbose:
        print()
        for name, passed, msg in _RESULTS:
            mark = "✓" if passed else "✗"
            print(f"  {mark} {name}")
            first = msg.split("\n")[0]
            print(f"      {first}")
            if not passed and "\n" in msg:
                for line in msg.split("\n")[1:6]:
                    print(f"      {line}")
        print(f"\n  {ok}/{len(_RESULTS)} 项通过，耗时 {time.time()-t0:.1f}s")
        if ok < len(_RESULTS):
            print("  ⚠ 有检查未通过——回测结果可能不可信，先修好再用。")
    return ok == len(_RESULTS)
