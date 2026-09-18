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


@check("策略：声明了精简指标集的，信号必须与全量指标下完全一致")
def _t_indicator_subset():
    """Strategy.indicators 声明少了不会报错——entry() 里的 KeyError 会被
    引擎/模拟盘/信号三处的 `except Exception: continue` 吞掉，该股票被静默跳过，
    表现为"策略不出信号"。这里直接调用，不给它被吞掉的机会。"""
    days = store.trading_days()
    if len(days) < 300:
        return "跳过：交易日不足"
    uni = universe.build(as_of=days[-1])
    if uni.empty:
        return "跳过：股票池为空"
    raw = store.load_daily(uni["code"].tolist()[:30], start=days[-400])

    checked = []
    for info in all_strategies():
        name = info["name"]
        s_min = get_strategy(name)
        decl = getattr(s_min, "indicators", None)
        if decl is None:
            continue
        unknown = set(decl) - set(ind.COMMON_COLUMNS)
        assert not unknown, f"{name} 声明了不存在的指标列 {sorted(unknown)}"

        s_full = get_strategy(name)
        s_full.indicators = None          # 实例级覆盖，退回全量指标
        n = 0
        for code, g in raw.groupby("code", sort=False):
            g = store.usable_history(g.sort_values("date"))
            if len(g) < 130:
                continue
            a, b = s_min.prepare(g), s_full.prepare(g)
            for fn in ("entry", "exit", "score"):
                x = np.asarray(getattr(s_min, fn)(a), dtype="float64")
                y = np.asarray(getattr(s_full, fn)(b), dtype="float64")
                assert x.shape == y.shape, f"{name}.{fn} 长度不同"
                assert np.allclose(x, y, equal_nan=True), (
                    f"{name}.{fn} 在精简指标集下与全量指标不一致——"
                    f"声明的 {sorted(decl)} 不够用")
            n += 1
        assert n >= 5, f"{name} 只比对到 {n} 只，样本太少，说明不了问题"
        checked.append(f"{name} {len(decl)}/{len(ind.COMMON_COLUMNS)} 列 × {n} 只")
    if not checked:
        return "没有策略声明精简指标集"
    return "；".join(checked)


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
# 自检跑在自己的账户里。早先的做法是整表备份再还原，但那有个要命的缺口：
# 自检中途被 Ctrl-C 或崩掉，用户的前向记录就停在「已清空、还没还原」的状态，
# 而那是不可再生的。跑在专属账户上，用户的记录压根不进入操作范围。
SELFTEST_ACCOUNT = "__selftest__"


def _paper_vs_engine(strategy: str, params: dict, cfg: engine.BacktestConfig,
                     fdf, codes, names, eng_start, start, end, top):
    """同一股票池、同一区间跑引擎与模拟盘，返回两边的成交明细。"""
    from . import paper
    from .data import universe as uni_mod

    st = get_strategy(strategy, **params)
    r = engine.run(st, codes, eng_start, end, cfg, names)

    orig = uni_mod.build
    uni_mod.build = lambda flt=None, as_of=None: fdf
    try:
        paper.reset(strategy, params, cfg.initial_cash, cfg.max_positions,
                    top, cfg.max_hold_days, account=SELFTEST_ACCOUNT)
        for d in store.trading_days(start=start, end=end):
            paper.advance(as_of=d, verbose=False, account=SELFTEST_ACCOUNT)
        with store.connect() as c:
            pt = pd.read_sql("SELECT code,open_date,close_date,shares,pnl "
                             "FROM paper_trade WHERE account=?", c,
                             params=(SELFTEST_ACCOUNT,))
    finally:
        uni_mod.build = orig

    et = pd.DataFrame([{"code": t.code, "open_date": t.open_date,
                        "close_date": t.close_date, "shares": t.shares,
                        "pnl": round(t.pnl, 2)} for t in r.trades])
    return et, pt


@check("模拟盘：与回测引擎逐笔等价（同一股票池、同一区间）")
def _t_paper_equiv():
    from . import paper
    from .data import universe as uni_mod

    days = store.trading_days()
    if len(days) < 40:
        return "跳过：交易日不足"
    # 残日必须排除在外。模拟盘会拒绝在残日上推进（照着几百只的残缺名单建仓
    # 是错的），引擎没有这层保护，两边自然对不上笔数。更要紧的是：在残日上
    # 比出来的「一致」本身没有意义——那是两边一起错得一样。
    complete = store.last_complete_day()
    if complete and complete in days:
        days = days[:days.index(complete) + 1]
    if len(days) < 40:
        return "跳过：完整交易日不足"
    end = days[-1]
    start = days[-31]                      # 最近约 30 个交易日
    eng_start = days[-32]                  # 引擎提前一日，使首个可交易日对齐

    fixed = uni_mod.build(as_of=start)
    if fixed.empty or len(fixed) < 50:
        return "跳过：股票池不足"
    codes = fixed["code"].tolist()[:150]
    names = dict(zip(fixed["code"], fixed["name"]))
    fdf = fixed[fixed["code"].isin(codes)]

    # 两个策略都要测：
    #   regime_momentum —— 常规策略，靠 exit() 离场
    #   growth_value    —— 打分型策略（score_fields + exit() 恒 False），
    #                      靠持有期上限调仓。这一类曾经在模拟盘里完全失效：
    #                      score() 是占位符返回 0，候选全同分退化成按代码序买；
    #                      又没有持有期概念，买满 5 只之后永远不动。两处都不报错。
    cases = [
        ("regime_momentum", {}, engine.BacktestConfig(
            initial_cash=200_000, max_positions=5)),
        ("growth_value", {}, engine.BacktestConfig(
            initial_cash=200_000, max_positions=5, max_hold_days=5)),
    ]

    notes = []
    try:
        for name, params, cfg in cases:
            et, pt = _paper_vs_engine(name, params, cfg, fdf, codes, names,
                                      eng_start, start, end, 150)
            if et.empty and pt.empty:
                notes.append(f"{name} 区间内无交易")
                continue
            assert len(et) == len(pt),                 f"{name} 成交笔数不同：引擎 {len(et)}，模拟盘 {len(pt)}"
            et = et.sort_values(["open_date", "code"]).reset_index(drop=True)
            pt = pt.sort_values(["open_date", "code"]).reset_index(drop=True)
            for col in ("code", "open_date", "close_date", "shares"):
                assert (et[col].values == pt[col].values).all(),                     f"{name} {col} 不一致"
            assert (abs(et["pnl"].values - pt["pnl"].values) < 0.05).all(),                 f"{name} 盈亏不一致"
            notes.append(f"{name} {len(et)} 笔一致")
    finally:
        # 只清自己的账户；用户的前向记录自始至终没被碰过
        paper.drop_account(SELFTEST_ACCOUNT)
    return "；".join(notes)


@check("模拟盘：多账户互不串台，且账户名非法时报错")
def _t_paper_accounts():
    """四张表原本是单账户的（主键不含 account）。迁移之后如果哪个查询漏了
    WHERE account=?，两个账户的持仓会混在一起——不报错，只是从此两个账户
    记的都不是自己的仓位，而前向记录一旦记错就没法回头重来。"""
    from . import paper

    a, b = "__selftest_a__", "__selftest_b__"
    try:
        paper.reset("ma_cross", {}, 100_000, 3, 50, 0, account=a)
        paper.reset("turtle_breakout", {}, 500_000, 5, 60, 10, account=b)

        sa, sb = paper.status(account=a), paper.status(account=b)
        assert sa["strategy"] == "ma_cross" and sb["strategy"] == "turtle_breakout", \
            f"策略串台：a={sa['strategy']} b={sb['strategy']}"
        assert sa["initial_cash"] == 100_000 and sb["initial_cash"] == 500_000, \
            f"本金串台：a={sa['initial_cash']} b={sb['initial_cash']}"
        assert sb["max_hold_days"] == 10 and sa["max_hold_days"] == 0, \
            "持有期上限串台"

        # 同一只票同时挂在两个账户下：老 schema 的 code 主键会在这里撞车
        with store.connect() as c:
            for acct, shares in ((a, 100), (b, 200)):
                c.execute("INSERT INTO paper_holding (account,code,name,shares,"
                          "cost,open_date,open_reason,peak,hold_days) "
                          "VALUES (?,?,?,?,?,?,?,?,0)",
                          (acct, "600000", "浦发银行", shares, 10.0,
                           "2026-01-05", "构造", 10.0))
        ha = paper._holdings(a)
        hb = paper._holdings(b)
        assert ha["600000"]["shares"] == 100 and hb["600000"]["shares"] == 200, \
            f"同票不同户读串了：a={ha['600000']['shares']} b={hb['600000']['shares']}"

        # 删掉 a 不能碰到 b
        paper.drop_account(a)
        assert not paper.status(account=a)["strategy"], "账户 a 未被删净"
        assert paper.status(account=b)["strategy"] == "turtle_breakout", \
            "删 a 把 b 一起删了"
        assert paper._holdings(b), "删 a 把 b 的持仓也删了"

        for bad in ("", "   ", "a" * 33, "a/b", "a;drop"):
            try:
                paper.check_name(bad)
            except ValueError:
                continue
            raise AssertionError(f"非法账户名 {bad!r} 应当被拒绝")
    finally:
        paper.drop_account(a)
        paper.drop_account(b)
    return "两账户隔离、同票不同户、删除不误伤、非法名被拒"


@check("模拟盘：老库（无 account 列）迁移后一行不丢，且可重复迁移")
def _t_paper_migration():
    """迁移只跑一次，跑错了就把不可再生的前向记录搞没了。所以在临时库上
    重演一遍：造老 schema、灌数据、迁移，逐字段比对。"""
    import sqlite3
    import tempfile

    from . import paper

    old = """
    CREATE TABLE paper_meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE paper_holding (
        code TEXT PRIMARY KEY, name TEXT, shares INTEGER, cost REAL,
        open_date TEXT, open_reason TEXT, peak REAL, hold_days INTEGER DEFAULT 0);
    CREATE TABLE paper_trade (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT, name TEXT, open_date TEXT, close_date TEXT, shares INTEGER,
        open_price REAL, close_price REAL, pnl REAL, pnl_pct REAL,
        hold_days INTEGER, open_reason TEXT, close_reason TEXT);
    CREATE TABLE paper_equity (
        date TEXT PRIMARY KEY, equity REAL, cash REAL, positions INTEGER,
        bench REAL, note TEXT);
    """
    seed = {
        "paper_meta": [("strategy", "growth_value"), ("cash", "4043.06"),
                       ("initial_cash", "200000.0"), ("last_date", "2026-08-17")],
        "paper_holding": [("600000", "浦发银行", 1000, 10.5,
                           "2026-08-10", "买入理由", 11.0, 3)],
        "paper_trade": [(1, "000001", "平安银行", "2026-07-01", "2026-07-20",
                         500, 12.0, 13.0, 480.5, 0.08, 13, "开仓", "平仓")],
        "paper_equity": [("2026-08-17", 203000.0, 4043.06, 5, 4100.5, "")],
    }

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/legacy.db"
        c = sqlite3.connect(path)
        try:
            c.executescript(old)
            for t, rows in seed.items():
                ph = ",".join("?" * len(rows[0]))
                c.executemany(f"INSERT INTO {t} VALUES ({ph})", rows)
            c.commit()

            for _ in range(3):                 # 幂等：迁移三次结果必须一样
                paper._migrate(c)
                for ddl in paper._DDL.values():
                    c.execute(ddl)
            c.commit()

            legacy = dict(paper._LEGACY_COLS,
                          paper_trade="id,code,name,open_date,close_date,shares,"
                                      "open_price,close_price,pnl,pnl_pct,"
                                      "hold_days,open_reason,close_reason")
            for t, rows in seed.items():
                cols = {r[1] for r in c.execute(f"PRAGMA table_info({t})")}
                assert "account" in cols, f"{t} 迁移后没有 account 列"
                got = c.execute(f"SELECT {legacy[t]} FROM {t} "
                                "ORDER BY rowid").fetchall()
                assert got == rows, f"{t} 数据在迁移中变了：{got} != {rows}"
                n = c.execute(f"SELECT COUNT(*) FROM {t} "
                              "WHERE account='default'").fetchone()[0]
                assert n == len(rows), \
                    f"{t} 有 {len(rows) - n} 行没归到 default 账户"

            assert not c.execute("SELECT name FROM sqlite_master "
                                 "WHERE name LIKE '%_old'").fetchall(), \
                "迁移留下了 _old 残表"

            # 上次迁移中途崩掉会留下 _old，再迁移必须能自愈而不是撞名报错
            c.execute("CREATE TABLE paper_meta_old (key TEXT, value TEXT)")
            c.commit()
            paper._migrate(c)
            c.commit()
            assert c.execute("SELECT COUNT(*) FROM paper_meta").fetchone()[0] == \
                len(seed["paper_meta"]), "自愈后 paper_meta 数据不完整"
        finally:
            c.close()
    return f"4 张表逐字段一致、迁移 3 次幂等、_old 残留可自愈"


@check("回测：结束日撞上库尾残日必须告警，默认结束日不能落在残日上")
def _t_backtest_tail():
    """库尾残日（同步中断或数据源部分失败留下的、只有几百只股票的交易日）
    一旦进了回测区间，当天没有 K 线的几千只票会被引擎判成停牌——不可买卖、
    持仓冻结计价。绩效数字照样出得来，只是最后那几天是假的，而且不报错。

    实测过一次：4 个数据源同时失败，库尾 6 天每天只剩 633 只沪市主板，
    而回测的默认结束日是「今天」，正好全踩上。
    """
    complete = store.last_complete_day()
    assert complete, "last_complete_day 不该为空"

    # 默认结束日：API 与 CLI 用的是同一个 store.last_complete_day()
    from .api import BacktestReq
    req = BacktestReq(strategy="ma_cross")
    assert req.end <= complete, (
        f"回测默认结束日 {req.end} 晚于库内最后一个完整交易日 {complete}，"
        "最后几天会落在残日上")

    days = store.trading_days()
    if len(days) < 60:
        return "跳过：交易日不足"

    uni = universe.build(as_of=days[-40])
    if uni.empty or len(uni) < 30:
        return f"默认结束日 {req.end} 已对齐；股票池不足，未验证告警"
    codes = uni["code"].tolist()[:30]
    names = dict(zip(uni["code"], uni["name"]))
    strat = get_strategy("ma_cross")
    cfg = engine.BacktestConfig(initial_cash=200_000, max_positions=3)
    start = days[-40]

    # 结束日对齐完整交易日时不该有任何告警
    clean = engine.run(strat, codes, start, complete, cfg, names)
    assert not clean.warnings, f"结束日已对齐 {complete}，不该告警：{clean.warnings}"

    if days[-1] == complete:
        return f"库尾无残日；默认结束日 {req.end} 已对齐，干净区间无告警"

    # 库里当前就有残日，把它拿进区间，必须告警
    dirty = engine.run(strat, codes, start, days[-1], cfg, names)
    assert dirty.warnings, (
        f"区间末尾 {days[-1]} 是残日（完整日只到 {complete}），却一条告警都没有")
    n_tail = sum(1 for d in store.trading_days(start=start, end=days[-1])
                 if d > complete)
    assert str(n_tail) in dirty.warnings[0], (
        f"告警里没说清有几天不完整：{dirty.warnings[0]}")
    return (f"默认结束日 {req.end} 已对齐；干净区间无告警；"
            f"含 {n_tail} 个残日的区间已告警")


@check("模拟盘：catch_up 必须逐日推进，不能跳过中间交易日")
def _t_catch_up_daily():
    """前向记录的价值在于逐日连续，中间缺几天就不再是「每天做了什么决策」的账。

    这里曾经是循环调用 advance()（不带 as_of），结果只推进一天就停：
    advance 省略 as_of 时直接取最后一个完整交易日，第一次就跳到最新那天并把
    last_date 设成它，第二次立刻被「已处理过」的守卫挡掉。中间交易日整段消失，
    而 last_date 照样显示「已追平到最新交易日」——从外面一点都看不出来。
    实测踩过：账户落后 5 个交易日，跑完只推进 1 天。
    """
    from . import paper

    complete = store.last_complete_day()
    assert complete, "last_complete_day 不该为空"
    days = store.trading_days(end=complete)
    if len(days) < 10:
        return "跳过：交易日不足"

    acct = "__selftest_catchup__"
    back = days[-5]                      # 制造「落后 4 个交易日」
    want = [d for d in days if d > back]
    try:
        paper.reset("ma_cross", {}, 100_000, 3, 40, 0, account=acct)
        paper._set("last_date", back, account=acct)
        assert paper.lag_days(acct) == len(want), (
            f"lag_days 说落后 {paper.lag_days(acct)} 天，实际应为 {len(want)}")

        n = paper.catch_up(verbose=False, account=acct)
        assert n == len(want), (
            f"落后 {len(want)} 个交易日，catch_up 只推进了 {n} 个")

        with store.connect() as c:
            got = [r[0] for r in c.execute(
                "SELECT date FROM paper_equity WHERE account=? ORDER BY date",
                (acct,)).fetchall()]
        assert got == want, f"净值记录有空洞：期望 {want}，实际 {got}"
        assert paper.lag_days(acct) == 0, "推完之后不该还落后"

        # 新账户（从未推进过）只处理最新那天，不回补历史
        fresh = "__selftest_fresh__"
        try:
            paper.reset("ma_cross", {}, 100_000, 3, 40, 0, account=fresh)
            assert paper.lag_days(fresh) == 0, (
                "新账户不该显示成落后——它从建立之后才开始记")
            m = paper.catch_up(verbose=False, account=fresh)
            assert m == 1, f"新账户应只推进最新一天，实际推进 {m} 天"
        finally:
            paper.drop_account(fresh)
    finally:
        paper.drop_account(acct)
    return f"落后 {len(want)} 天逐日补齐无空洞；新账户只推进 1 天"


@check("同步：--behind 只拉落后的股票，不把已经最新的几千只重扫一遍")
def _t_sync_behind():
    """默认的增量同步为了覆盖盘中写下的残缺 K 线，会把**每一只**都往回重拉 7 天。
    数据断过几天再来补时，绝大多数请求是白发的——实测断 3 天后，4599 只里
    3397 只已经是最新的，74% 的请求纯属浪费，还把限流额度提前打光，
    真正缺的那 1202 只反而补不上。

    这里用 cancel_check 让同步在发第一个请求前就返回，只比对它算出来的待更新
    名单，不碰网络。
    """
    from .data import source

    complete = store.last_complete_day()
    if not complete:
        return "跳过：没有完整交易日"

    def stop():
        return True

    plain = source.sync_daily(cancel_check=stop)
    behind = source.sync_daily(cancel_check=stop, behind_only=True)

    inst = store.load_instruments()
    alive = inst[inst["status"].fillna("listed") != "delisted"]["code"].tolist()
    last = store.last_dates()
    want = len([c for c in alive if last.get(c, "") < complete])

    # 不做精确相等：后台跑着同步时 last_dates 每秒都在变，算期望值和 sync_daily
    # 算待更新名单不在同一瞬间，实测差过 32 只。那种假失败比不检查更糟。
    # 这里校验的是数量级关系，足以抓住「--behind 没生效」这个真问题。
    assert behind["pending"] <= plain["pending"], (
        f"--behind 反而比默认拉得多：{behind['pending']} > {plain['pending']}")
    assert behind["pending"] <= want * 1.5 + 100, (
        f"--behind 待更新 {behind['pending']} 只，远超真正落后的 {want} 只——"
        "跳过逻辑多半没生效")
    if plain["pending"] > want + 200:
        assert behind["pending"] < plain["pending"], (
            f"有 {plain['pending'] - want} 只已是最新却没被 --behind 跳过")

    # 退市股不该出现在任何一边的待更新名单里（免费源永远取不到）
    assert plain["requested"] == len(alive), (
        f"日常同步请求了 {plain['requested']} 只，在册非退市只有 {len(alive)} 只——"
        "退市股又混进来了")

    saved = plain["pending"] - behind["pending"]
    return (f"默认 {plain['pending']} 只 -> --behind {behind['pending']} 只，"
            f"省掉 {saved} 次请求")


@check("打分：打分型策略在模拟盘与每日信号里都必须真正打分")
def _t_score_spread():
    """score_fields 类策略的 score() 是占位符，真正的横截面合成要在
    引擎、模拟盘、每日信号三处各做一次。漏掉任何一处都不报错，
    只会让「按打分取前 N 名」静默退化成「按代码序取前 N 名」。"""
    from . import paper

    st = get_strategy("growth_value")
    fields = list(getattr(st, "score_fields", None) or [])
    assert fields, "growth_value 应当声明 score_fields"

    # 两个分量不能构造成完全反相关，否则百分位相加恒为常数——那是构造出来的
    # 退化情形，验不出打分是否生效。用互质步长打散第二个分量。
    sig = {f"{i:06d}": {"fields": {fields[0][0]: float(i),
                                   fields[1][0]: float((i * 7) % 40)}}
           for i in range(40)}
    paper._apply_score_fields(sig, fields)
    vals = sorted(v["score"] for v in sig.values())
    assert len(set(round(v, 6) for v in vals)) > 1,         "候选分数全部相同——打分没有真正生效"
    assert 0.0 <= vals[0] and vals[-1] <= 1.0,         f"归一后应落在 [0,1]，实际 {vals[0]:.3f}~{vals[-1]:.3f}"
    n_distinct = len(set(round(v, 6) for v in vals))

    # 每日信号是实际据以决策的那一页，端到端跑一遍。
    # 收紧流动性下限只是为了让这项检查跑得快，不影响判据。
    from . import signals as sig_mod
    res = sig_mod.generate(get_strategy("growth_value"),
                           universe.UniverseFilter(min_amount=3e9),
                           sig_mod.SignalConfig(max_candidates=20))
    picks = res.get("buys") or []
    if len(picks) < 5:
        return f"合成正确（{n_distinct} 个不同分数）；每日信号候选不足，未端到端验证"
    scores = [b["score"] for b in picks]
    assert len(set(scores)) > 1,         f"每日信号的 {len(picks)} 个候选分数全部相同——打分没有生效，等于按代码序排"
    codes = [b["code"] for b in picks]
    assert codes != sorted(codes),         "每日信号候选恰好按代码升序，几乎可以肯定是没有真正排序"
    return (f"合成 {n_distinct} 个不同分数；"
            f"每日信号 {len(picks)} 个候选、{len(set(scores))} 个不同分数")


@check("股票池：历史不足时必须报错，不能静默跳过流动性过滤")
def _t_universe_loud():
    """曾经 as_of 早于库内最早交易日时，build() 会静默返回全市场未过滤名单，
    且未按成交额排序——「流动性前 N 只」实际取到的是代码序前 N 只。"""
    days = store.trading_days()
    if not days:
        return "跳过：本地还没有数据"
    too_early = (pd.Timestamp(days[0]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        uni = universe.build(as_of=too_early)
    except universe.InsufficientHistory:
        pass
    else:
        raise AssertionError(
            f"as_of={too_early} 早于库内最早交易日 {days[0]}，"
            f"却静默返回了 {len(uni)} 只，没有报错")

    ok = universe.build(as_of=days[min(60, len(days) - 1)])
    if ok.empty:
        return "报错正常；正常区间池子为空（数据太少）"
    amt = ok["avg_amount"]
    assert amt.notna().all(), "正常区间的 avg_amount 不应为空"
    assert (amt.diff().dropna() <= 1e-6).all(), "股票池必须按成交额降序排列"
    return f"历史不足已报错；正常区间 {len(ok)} 只且按成交额降序"


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


@check("数据：库尾的残日必须被挡在信号与模拟盘之外")
def _t_partial_tail():
    """同步中断会在库尾留下只有几只股票的残日。它照样是 trading_days() 的最后一项，
    于是几千只被判成当日停牌，候选静默缩成几只——不报错。
    实测 2026-08-12 库里只有 6 行，前一交易日 4577 行。"""
    from . import signals as sig_mod

    with store.connect() as c:
        rows = c.execute("SELECT date, COUNT(*) FROM daily GROUP BY date "
                         "ORDER BY date DESC LIMIT 30").fetchall()
    if len(rows) < 5:
        return "跳过：交易日不足"
    counts = sorted(n for _, n in rows)
    med = counts[len(counts) // 2]

    complete = store.last_complete_day()
    assert complete is not None, "last_complete_day 不应为空"
    got = dict(rows)[complete]
    assert got >= med * 0.5,         f"选中的 {complete} 只有 {got} 行，不到近期中位数 {med} 的一半"

    used, _ = sig_mod._latest_usable_date()
    assert used is None or dict(rows).get(used, med) >= med * 0.5,         f"每日信号用了残日 {used}（{dict(rows).get(used)} 行，中位数 {med}）"

    # 回补路径（--since / 建账户）用显式 as_of 推进，曾经绕过上面这层保护：
    # trading_days() 里库尾那几个残日会被一并推进，照残缺名单建仓，不报错。
    # 保护挡在 advance() 内部，所以这里直接拿残日去敲它。
    tail = rows[0][0]
    if tail != complete:
        from . import paper
        acct = "__selftest_tail__"
        try:
            paper.reset("ma_cross", {}, 100_000, 3, 50, 0, account=acct)
            ev = paper.advance(as_of=tail, verbose=False, account=acct)
            assert ev.get("skipped"), (
                f"模拟盘接受了残日 {tail}（{rows[0][1]} 行，中位数 {med}），"
                "会照着残缺的候选名单建仓")
        finally:
            paper.drop_account(acct)
        return (f"库尾 {tail} 只有 {rows[0][1]} 行，已回退到 {complete}（{got} 行）；"
                "模拟盘也拒绝了显式推进到该日")
    return f"库尾 {tail} 完整（{got} 行，中位数 {med}）"


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
    # 库尾可能是同步中断留下的残日，那种情况下回退是对的，由上一项检查负责
    complete = store.last_complete_day()
    if complete and complete != days[-1]:
        assert skipped and as_of == complete, (
            f"库尾 {days[-1]} 是残日，应回退到 {complete}，实际 as_of={as_of}")
        return f"库尾是残日，已回退到 {as_of}"
    assert as_of == days[-1] and not skipped, (
        f"已收盘且库尾完整，应当直接用 {days[-1]}，"
        f"实际 as_of={as_of}、skipped={skipped}")
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
