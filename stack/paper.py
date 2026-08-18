"""前向模拟盘：逐日推进的虚拟账户，把每天的决策落库。

和回测的区别在于**不可回溯**：每天的买卖在当天记录、之后不再改动。
回测可以反复重跑、反复调参，所以它证明不了策略有效；模拟盘的记录是一次性的，
半年后回看时，那就是真正没被人看过的未来数据。

撮合口径与回测引擎完全一致：
  前一交易日收盘产生信号 → 当日开盘成交 → 当日收盘计价
  涨跌停买不进/卖不出、停牌不可交易、按 A 股费率扣费、100 股整手。
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .config import LOT_SIZE, buy_cost, price_limit, sell_cost
from .data import store, universe
from .strategies import get_strategy

EPS = 1e-6

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS paper_holding (
    code TEXT PRIMARY KEY, name TEXT, shares INTEGER, cost REAL,
    open_date TEXT, open_reason TEXT, peak REAL, hold_days INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS paper_trade (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT, name TEXT, open_date TEXT, close_date TEXT, shares INTEGER,
    open_price REAL, close_price REAL, pnl REAL, pnl_pct REAL,
    hold_days INTEGER, open_reason TEXT, close_reason TEXT
);

CREATE TABLE IF NOT EXISTS paper_equity (
    date TEXT PRIMARY KEY, equity REAL, cash REAL, positions INTEGER,
    bench REAL, note TEXT
);
"""


def _init():
    with store.connect() as c:
        c.executescript(SCHEMA)


def _meta(k, default=None):
    with store.connect() as c:
        r = c.execute("SELECT value FROM paper_meta WHERE key=?", (k,)).fetchone()
    return r[0] if r else default


def _set(k, v):
    with store.connect() as c:
        c.execute("INSERT INTO paper_meta (key,value) VALUES (?,?) "
                  "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, str(v)))


def reset(strategy: str = "regime_momentum", params: dict | None = None,
          cash: float = 200_000, max_positions: int = 5,
          top: int = 400, max_hold_days: int = 0) -> dict:
    """新建/重置模拟盘。会清空已有记录。"""
    _init()
    get_strategy(strategy, **(params or {}))      # 先校验参数合法
    with store.connect() as c:
        for t in ("paper_holding", "paper_trade", "paper_equity", "paper_meta"):
            c.execute(f"DELETE FROM {t}")
    for k, v in [("strategy", strategy), ("params", json.dumps(params or {})),
                 ("cash", cash), ("initial_cash", cash),
                 ("max_positions", max_positions), ("top", top),
                 ("max_hold_days", max_hold_days),
                 ("last_date", ""), ("created_at",
                                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]:
        _set(k, v)
    return status()


def _apply_score_fields(sig: dict, fields: list) -> None:
    """多因子打分：逐字段做横截面百分位归一后加权相加，与回测引擎口径一致。

    这一步必须在拿到全部候选之后做。策略的 score() 只看得到单只股票的时序，
    在那里排序排的是时间维度、还会用到未来数据——所以 score_fields 类策略的
    score() 都是返回 0 的占位符，真正的合成在引擎（回测）和这里（模拟盘）。

    曾经这里直接用 strategy.score()，于是 growth_value 的候选全是 0 分，
    "按打分取前 N 名"静默退化成"按代码序取前 N 名"，不报错、也看不出来。
    """
    codes = list(sig)
    if not codes:
        return
    total = pd.Series(0.0, index=codes, dtype="float64")
    wsum = 0.0
    for col, w in fields:
        v = pd.Series({c: sig[c]["fields"].get(col, float("nan")) for c in codes},
                      dtype="float64")
        # 与引擎 _cross_rank 一致：有效值归一到 [0,1]，缺失记 0 分
        pct = (v.rank(method="average") - 1) / max(v.notna().sum() - 1, 1)
        total += w * pct.fillna(0.0)
        wsum += w
    if wsum:
        total /= wsum
    for c in codes:
        sig[c]["score"] = float(total[c])


def _holdings() -> dict:
    with store.connect() as c:
        rows = c.execute("SELECT code,name,shares,cost,open_date,open_reason,"
                         "peak,hold_days FROM paper_holding").fetchall()
    return {r[0]: {"name": r[1], "shares": r[2], "cost": r[3], "open_date": r[4],
                   "open_reason": r[5], "peak": r[6], "hold_days": r[7]}
            for r in rows}


def advance(as_of: str | None = None, verbose: bool = True) -> dict:
    """推进一个交易日。返回当日发生的事情。

    as_of 省略时取本地库里最后一个**已收盘**的交易日。已处理过的日期会跳过，
    所以重复运行是安全的，不会把同一天算两遍。
    """
    _init()
    if not _meta("strategy"):
        raise RuntimeError("模拟盘还没初始化，先运行：stack.cli paper init")

    from .data import source
    days = store.trading_days()
    if not days:
        raise RuntimeError("本地没有行情数据")

    if as_of is None:
        as_of = days[-1]
        today = datetime.now().strftime("%Y-%m-%d")
        if as_of == today and not source.market_closed_today():
            as_of = days[-2] if len(days) > 1 else None
    if as_of is None:
        return {"skipped": "没有已收盘的交易日"}

    last = _meta("last_date", "")
    if last and as_of <= last:
        return {"skipped": f"{as_of} 已处理过（最后处理到 {last}）"}

    i = days.index(as_of)
    if i == 0:
        return {"skipped": "缺少前一交易日"}
    prev = days[i - 1]

    strat = get_strategy(_meta("strategy"), **json.loads(_meta("params", "{}")))
    cash = float(_meta("cash"))
    max_pos = int(_meta("max_positions"))
    top = int(_meta("top"))
    max_hold = int(_meta("max_hold_days", 0) or 0)
    holds = _holdings()

    # ---------------- 数据：股票池 + 持仓，窗口够算指标即可 ----------------
    uni = universe.build(as_of=prev)
    codes = set(uni["code"].tolist()[:top]) | set(holds)
    names = dict(zip(uni["code"], uni["name"]))
    need = max(600, int(strat.warmup_bars() * 1.5) + 60)
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=need)).strftime("%Y-%m-%d")
    raw = store.load_daily(codes=sorted(codes), start=start, end=as_of)

    score_fields = list(getattr(strat, "score_fields", None) or [])
    bars, sig = {}, {}
    for code, g in raw.groupby("code", sort=False):
        g = store.usable_history(g.sort_values("date"))
        if len(g) < 130:
            continue
        idx = g["date"].dt.strftime("%Y-%m-%d")
        bars[code] = g.set_index(idx)
        if prev not in bars[code].index:
            continue
        try:
            d = strat.prepare(g)
            d.index = idx
            sig[code] = {"entry": bool(strat.entry(d).loc[prev]),
                         "exit": bool(strat.exit(d).loc[prev]),
                         "score": float(strat.score(d).loc[prev]),
                         "row": d.loc[prev],
                         "fields": {col: float(d.loc[prev, col])
                                    for col, _w in score_fields
                                    if col in d.columns}}
        except Exception:
            continue

    if score_fields:
        _apply_score_fields(sig, score_fields)

    # 大盘择时
    reg = strat.market_regime(days[:i])
    regime_on = True if reg is None else bool(reg.iloc[-1])

    events = {"date": as_of, "sells": [], "buys": [], "blocked": [],
              "regime_on": regime_on}

    def px_at(code, date, field):
        b = bars.get(code)
        if b is None or date not in b.index:
            return None
        v = float(b.loc[date, field])
        return v if v > 0 else None

    # ---------------- 1. 卖出 ----------------
    for code in list(holds):
        h = holds[code]
        h["hold_days"] += 1
        s = sig.get(code)
        o, pc = px_at(code, as_of, "open"), px_at(code, prev, "close")
        if s is None or o is None or pc is None:
            events["blocked"].append(f"{code} 停牌或数据缺失，无法处理")
            continue
        if h["hold_days"] < 1:
            continue
        # 到期调仓：growth_value 这类策略 exit() 恒为 False，靠持有期上限重排。
        # 没有这一条，它会买满仓位后永远不动——不报错，只是从此不再是那个策略。
        expired = max_hold > 0 and h["hold_days"] >= max_hold
        if not s["exit"] and not expired:
            continue
        if o <= pc * (1 - price_limit(code, h["name"])) + EPS:
            events["blocked"].append(f"{code} {h['name']} 开盘跌停，卖不出")
            continue
        gross = o * h["shares"]
        proceeds = gross - sell_cost(gross)
        cash += proceeds
        pnl = proceeds - h["cost"] * h["shares"]
        reason = (f"持有满 {max_hold} 日到期" if expired and not s["exit"]
                  else strat.reason(s["row"], "SELL"))
        with store.connect() as c:
            c.execute("""INSERT INTO paper_trade (code,name,open_date,close_date,shares,
                open_price,close_price,pnl,pnl_pct,hold_days,open_reason,close_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (code, h["name"], h["open_date"], as_of, h["shares"],
                       round(h["cost"], 3), round(o, 3), round(pnl, 2),
                       round(pnl / (h["cost"] * h["shares"]), 4), h["hold_days"],
                       h["open_reason"], reason))
            c.execute("DELETE FROM paper_holding WHERE code=?", (code,))
        events["sells"].append({"code": code, "name": h["name"], "price": round(o, 2),
                                "shares": h["shares"], "pnl": round(pnl, 2),
                                "pnl_pct": round(pnl / (h["cost"] * h["shares"]), 4),
                                "reason": reason})
        del holds[code]

    # ---------------- 2. 买入 ----------------
    slots = max_pos - len(holds)
    if regime_on and slots > 0:
        cands = [(v["score"], c) for c, v in sig.items()
                 if v["entry"] and c not in holds]
        cands.sort(key=lambda x: -x[0] if x[0] == x[0] else 9e9)
        mv = sum(h["shares"] * (px_at(c, as_of, "close") or h["cost"])
                 for c, h in holds.items())
        budget = (cash + mv) / max_pos
        for _, code in cands:
            if slots <= 0:
                break
            o, pc = px_at(code, as_of, "open"), px_at(code, prev, "close")
            if o is None or pc is None:
                continue
            name = names.get(code, code)
            if o >= pc * (1 + price_limit(code, name)) - EPS:
                events["blocked"].append(f"{code} {name} 开盘涨停，买不进")
                continue
            shares = int(budget / o // LOT_SIZE) * LOT_SIZE
            while shares > 0 and o * shares + buy_cost(o * shares) > cash:
                shares -= LOT_SIZE
            if shares <= 0:
                events["blocked"].append(f"{code} {name} 资金不足一手")
                continue
            gross = o * shares
            fee = buy_cost(gross)
            cash -= gross + fee
            cost = (gross + fee) / shares
            reason = strat.reason(sig[code]["row"], "BUY")
            with store.connect() as c:
                c.execute("""INSERT INTO paper_holding
                    (code,name,shares,cost,open_date,open_reason,peak,hold_days)
                    VALUES (?,?,?,?,?,?,?,0)""",
                          (code, name, shares, cost, as_of, reason, cost))
            holds[code] = {"name": name, "shares": shares, "cost": cost,
                           "open_date": as_of, "open_reason": reason,
                           "peak": cost, "hold_days": 0}
            events["buys"].append({"code": code, "name": name, "price": round(o, 2),
                                   "shares": shares, "amount": round(gross + fee, 2),
                                   "reason": reason})
            slots -= 1

    # ---------------- 3. 盯市 ----------------
    mv = 0.0
    for code, h in holds.items():
        c_px = px_at(code, as_of, "close") or h["cost"]
        h["peak"] = max(h["peak"], c_px)
        mv += h["shares"] * c_px
        with store.connect() as conn:
            conn.execute("UPDATE paper_holding SET peak=?, hold_days=? WHERE code=?",
                         (h["peak"], h["hold_days"], code))
    equity = cash + mv

    bench = None
    bd = store.load_daily(["IDX000300"], end=as_of)
    if not bd.empty:
        bench = float(bd.sort_values("date")["close"].iloc[-1])

    with store.connect() as c:
        c.execute("""INSERT OR REPLACE INTO paper_equity
                     (date,equity,cash,positions,bench,note) VALUES (?,?,?,?,?,?)""",
                  (as_of, round(equity, 2), round(cash, 2), len(holds), bench,
                   "" if regime_on else "择时空仓"))
    _set("cash", cash)
    _set("last_date", as_of)

    events["equity"] = round(equity, 2)
    events["cash"] = round(cash, 2)
    events["positions"] = len(holds)
    return events


def catch_up(max_days: int = 400, verbose: bool = True) -> int:
    """从上次处理到的地方一路推进到最新已收盘交易日。"""
    n = 0
    for _ in range(max_days):
        ev = advance()
        if ev.get("skipped"):
            break
        n += 1
        if verbose and (ev["buys"] or ev["sells"]):
            print(f"  {ev['date']}  买{len(ev['buys'])} 卖{len(ev['sells'])}  "
                  f"净值 {ev['equity']:,.0f}")
    return n


def status() -> dict:
    _init()
    init_cash = float(_meta("initial_cash", 0) or 0)
    with store.connect() as c:
        eq = pd.read_sql("SELECT * FROM paper_equity ORDER BY date", c)
        hold = pd.read_sql("SELECT * FROM paper_holding", c)
        tr = pd.read_sql("SELECT * FROM paper_trade ORDER BY close_date", c)
    out = {"strategy": _meta("strategy"), "params": json.loads(_meta("params", "{}")),
           "initial_cash": init_cash, "cash": float(_meta("cash", 0) or 0),
           "max_positions": int(_meta("max_positions", 5) or 5),
           "max_hold_days": int(_meta("max_hold_days", 0) or 0),
           "created_at": _meta("created_at"), "last_date": _meta("last_date"),
           "days": len(eq), "holdings": hold.to_dict("records"),
           "closed_trades": len(tr)}
    if not eq.empty:
        e = eq["equity"].to_numpy()
        out["equity"] = float(e[-1])
        out["total_return"] = float(e[-1] / init_cash - 1) if init_cash else 0.0
        out["max_drawdown"] = float((e / pd.Series(e).cummax().to_numpy() - 1).min())
        b = eq["bench"].dropna()
        if len(b) > 1:
            out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        out["idle_days"] = int((eq["note"] == "择时空仓").sum())
    if not tr.empty:
        out["win_rate"] = float((tr["pnl"] > 0).mean())
        out["realized_pnl"] = float(tr["pnl"].sum())
    return out
