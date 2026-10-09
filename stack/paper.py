"""前向模拟盘：逐日推进的虚拟账户，把每天的决策落库。

和回测的区别在于**不可回溯**：每天的买卖在当天记录、之后不再改动。
回测可以反复重跑、反复调参，所以它证明不了策略有效；模拟盘的记录是一次性的，
半年后回看时，那就是真正没被人看过的未来数据。

撮合口径与回测引擎完全一致：
  前一交易日收盘产生信号 → 当日开盘成交 → 当日收盘计价
  涨跌停买不进/卖不出、停牌不可交易、按 A 股费率扣费、100 股整手。

**多账户**：前向验证的成本是时间，只验一个策略意味着半年后只得到一个样本，
而且它不行的话这半年就白等了。并行跑几个的边际成本几乎为零——同一份数据、
同一天推进——所以四张表都带 account 维度，各账户互不干扰。
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd

from .config import (LOT_SIZE, buy_cost, hit_limit_down, hit_limit_up,
                     price_limit, sell_cost)
from .data import store, universe
from .strategies import get_strategy
from .strategies.base import blend_score_fields

DEFAULT_ACCOUNT = "default"
_NAME_RE = re.compile(r"^[\w一-龥-]{1,32}$")

# 每张表单独一条 DDL：迁移时要能只重建其中一张，整段 executescript 做不到这点。
_DDL = {
    "paper_meta": """
CREATE TABLE IF NOT EXISTS paper_meta (
    account TEXT NOT NULL DEFAULT 'default',
    key     TEXT NOT NULL,
    value   TEXT,
    PRIMARY KEY (account, key)
)""",
    "paper_holding": """
CREATE TABLE IF NOT EXISTS paper_holding (
    account TEXT NOT NULL DEFAULT 'default',
    code TEXT NOT NULL, name TEXT, shares INTEGER, cost REAL,
    open_date TEXT, open_reason TEXT, peak REAL, hold_days INTEGER DEFAULT 0,
    PRIMARY KEY (account, code)
)""",
    "paper_trade": """
CREATE TABLE IF NOT EXISTS paper_trade (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL DEFAULT 'default',
    code TEXT, name TEXT, open_date TEXT, close_date TEXT, shares INTEGER,
    open_price REAL, close_price REAL, pnl REAL, pnl_pct REAL,
    hold_days INTEGER, open_reason TEXT, close_reason TEXT
)""",
    "paper_equity": """
CREATE TABLE IF NOT EXISTS paper_equity (
    account TEXT NOT NULL DEFAULT 'default',
    date TEXT NOT NULL, equity REAL, cash REAL, positions INTEGER,
    bench REAL, note TEXT,
    PRIMARY KEY (account, date)
)""",
}

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_paper_trade_acct "
    "ON paper_trade (account, close_date)",
)

# 老库（单账户）里各表的列，顺序必须与重建时的 INSERT ... SELECT 一致
_LEGACY_COLS = {
    "paper_meta": "key,value",
    "paper_holding": "code,name,shares,cost,open_date,open_reason,peak,hold_days",
    "paper_equity": "date,equity,cash,positions,bench,note",
}


def _table_cols(c, table: str) -> set:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}


def _migrate(c) -> None:
    """老库平滑升级：四张表原本没有 account 列（全局单账户）。

    sqlite 改不了主键，所以 meta/holding/equity 三张只能重建。老数据整体归入
    'default' 账户——模拟盘记录是不可再生的前向证据，迁移丢一行都不行。
    幂等：跑过之后再跑什么都不做。
    """
    # paper_trade 靠自增 id 做主键，不用重建，加列即可
    have = _table_cols(c, "paper_trade")
    if have and "account" not in have:
        c.execute("ALTER TABLE paper_trade ADD COLUMN account TEXT "
                  "NOT NULL DEFAULT 'default'")

    for t, cols in _LEGACY_COLS.items():
        # 上一次迁移中途崩了会留下 _old，先清掉再重来，否则 RENAME 会撞名
        c.execute(f"DROP TABLE IF EXISTS {t}_old")
        have = _table_cols(c, t)
        if not have or "account" in have:
            continue
        c.execute(f"ALTER TABLE {t} RENAME TO {t}_old")
        c.execute(_DDL[t])
        c.execute(f"INSERT INTO {t} (account,{cols}) "
                  f"SELECT '{DEFAULT_ACCOUNT}',{cols} FROM {t}_old")
        c.execute(f"DROP TABLE {t}_old")


def _init():
    with store.connect() as c:
        _migrate(c)
        for ddl in _DDL.values():
            c.execute(ddl)
        for idx in _INDEXES:
            c.execute(idx)


def check_name(account: str) -> str:
    """账户名限中英文/数字/下划线/连字符。

    也顺带挡住空名字——否则它会静默变成 default，把两个账户的记录混在一起。
    """
    account = (account or "").strip()
    if not _NAME_RE.match(account):
        raise ValueError(f"账户名不合法：{account!r}（限 1-32 位中英文/数字/_/-）")
    return account


def _meta(k, default=None, account: str = DEFAULT_ACCOUNT):
    with store.connect() as c:
        r = c.execute("SELECT value FROM paper_meta WHERE account=? AND key=?",
                      (account, k)).fetchone()
    return r[0] if r else default


def _set(k, v, account: str = DEFAULT_ACCOUNT):
    with store.connect() as c:
        c.execute("INSERT INTO paper_meta (account,key,value) VALUES (?,?,?) "
                  "ON CONFLICT(account,key) DO UPDATE SET value=excluded.value",
                  (account, k, str(v)))


def list_accounts(include_internal: bool = False) -> list[dict]:
    """所有账户的概览。default 排最前，其余按创建时间。

    `__` 开头的是内部账户（自检用），默认不列出：否则界面上会冒出来，
    而且服务里的守护线程会去推进它们——自检正拿显式日期在同一个账户上推，
    两边一撞，自检就会报出莫名其妙的失败。
    """
    _init()
    with store.connect() as c:
        rows = c.execute(
            "SELECT account, MAX(CASE WHEN key='strategy' THEN value END),"
            "       MAX(CASE WHEN key='created_at' THEN value END),"
            "       MAX(CASE WHEN key='last_date' THEN value END) "
            "FROM paper_meta GROUP BY account").fetchall()
    out = [{"account": r[0], "strategy": r[1], "created_at": r[2],
            "last_date": r[3] or ""} for r in rows
           if r[1] and (include_internal or not r[0].startswith("__"))]
    out.sort(key=lambda x: (x["account"] != DEFAULT_ACCOUNT, x["created_at"] or ""))
    return out


def drop_account(account: str) -> dict:
    """删除一个账户的全部记录。前向记录不可再生，删了就没了。"""
    account = check_name(account)
    _init()
    removed = {}
    with store.connect() as c:
        for t in ("paper_holding", "paper_trade", "paper_equity", "paper_meta"):
            cur = c.execute(f"DELETE FROM {t} WHERE account=?", (account,))
            removed[t] = cur.rowcount
    return {"account": account, "removed": removed}


def reset(strategy: str = "regime_momentum", params: dict | None = None,
          cash: float = 200_000, max_positions: int = 5,
          top: int = 400, max_hold_days: int = 0,
          account: str = DEFAULT_ACCOUNT, renew_ranked: bool = False) -> dict:
    """新建/重置一个账户。只清空该账户，其他账户不受影响。

    renew_ranked：到期仍在前 max_positions 名就续持，见 BacktestConfig.renew_ranked。
    """
    account = check_name(account)
    _init()
    get_strategy(strategy, **(params or {}))      # 先校验参数合法
    with store.connect() as c:
        for t in ("paper_holding", "paper_trade", "paper_equity", "paper_meta"):
            c.execute(f"DELETE FROM {t} WHERE account=?", (account,))
    for k, v in [("strategy", strategy), ("params", json.dumps(params or {})),
                 ("cash", cash), ("initial_cash", cash),
                 ("max_positions", max_positions), ("top", top),
                 ("max_hold_days", max_hold_days),
                 ("renew_ranked", int(bool(renew_ranked))),
                 ("last_date", ""), ("created_at",
                                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"))]:
        _set(k, v, account=account)
    return status(account=account)


def _apply_score_fields(sig: dict, fields: list) -> None:
    """把横截面合成的分数写回候选表。合成口径见 strategies.base.blend_score_fields。"""
    scores = blend_score_fields({c: v["fields"] for c, v in sig.items()}, fields)
    for c, v in scores.items():
        sig[c]["score"] = v


def _holdings(account: str = DEFAULT_ACCOUNT) -> dict:
    with store.connect() as c:
        rows = c.execute("SELECT code,name,shares,cost,open_date,open_reason,"
                         "peak,hold_days FROM paper_holding WHERE account=?",
                         (account,)).fetchall()
    return {r[0]: {"name": r[1], "shares": r[2], "cost": r[3], "open_date": r[4],
                   "open_reason": r[5], "peak": r[6], "hold_days": r[7]}
            for r in rows}


def advance(as_of: str | None = None, verbose: bool = True,
            account: str = DEFAULT_ACCOUNT) -> dict:
    """推进一个交易日。返回当日发生的事情。

    as_of 省略时取本地库里最后一个**已收盘**的交易日。已处理过的日期会跳过，
    所以重复运行是安全的，不会把同一天算两遍。
    """
    _init()
    if not _meta("strategy", account=account):
        raise RuntimeError(f"账户 {account} 还没初始化，先运行：stack.cli paper init")

    from .data import source
    days = store.trading_days()
    if not days:
        raise RuntimeError("本地没有行情数据")

    if as_of is None:
        as_of = days[-1]
        today = datetime.now().strftime("%Y-%m-%d")
        if as_of == today and not source.market_closed_today():
            as_of = days[-2] if len(days) > 1 else None
        # 残日（同步中断留下的几行）不能拿来建仓：其余几千只会被判成停牌，
        # 模拟盘会照着一份残缺的候选名单下单，而且不报错。
        complete = store.last_complete_day()
        if as_of and complete and as_of > complete:
            as_of = complete
    else:
        # 显式指定日期的路径（回补、自检）同样会踩到残日：`--since` 回补会把
        # trading_days() 里库尾那几个只有几百行的日子一并推进，照着一份残缺的
        # 候选名单建仓，不报错。挡在这里而不是挡在每个调用方，是因为漏掉一个
        # 调用方不会有任何症状——前向记录会照样写下来，只是内容是错的。
        complete = store.last_complete_day()
        if complete and as_of > complete:
            return {"skipped": f"{as_of} 数据不完整（库内最后完整交易日 "
                               f"{complete}），不能拿残日建仓"}
    if as_of is None:
        return {"skipped": "没有已收盘的交易日"}

    last = _meta("last_date", "", account=account)
    if last and as_of <= last:
        return {"skipped": f"{as_of} 已处理过（最后处理到 {last}）"}

    i = days.index(as_of)
    if i == 0:
        return {"skipped": "缺少前一交易日"}
    prev = days[i - 1]

    strat = get_strategy(_meta("strategy", account=account),
                         **json.loads(_meta("params", "{}", account=account)))
    cash = float(_meta("cash", account=account))
    max_pos = int(_meta("max_positions", account=account))
    top = int(_meta("top", account=account))
    max_hold = int(_meta("max_hold_days", 0, account=account) or 0)
    # 老账户没有这一项：按 0 处理，行为与加这个开关之前完全一致
    renew_ranked = bool(int(_meta("renew_ranked", 0, account=account) or 0))
    holds = _holdings(account)

    # ---------------- 数据：股票池 + 持仓，窗口够算指标即可 ----------------
    uni = universe.build(as_of=prev)
    codes = set(uni["code"].tolist()[:top]) | set(holds)
    names = dict(zip(uni["code"], uni["name"]))
    need = max(600, int(strat.warmup_bars() * 1.5) + 60)
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=need)).strftime("%Y-%m-%d")
    raw = store.load_daily(codes=sorted(codes), start=start, end=as_of)

    score_fields = list(getattr(strat, "score_fields", None) or [])
    bars, sig = {}, {}
    for code, g in store.iter_stocks(raw):
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

    events = {"date": as_of, "account": account, "sells": [], "buys": [],
              "blocked": [], "regime_on": regime_on}

    def px_at(code, date, field):
        b = bars.get(code)
        if b is None or date not in b.index:
            return None
        v = float(b.loc[date, field])
        return v if v > 0 else None

    def last_close(code, fallback):
        """as_of 及之前最后一根 K 线的收盘价。停牌的持仓按它计价，与引擎的
        _Position.last 一致；原来按成本价，停牌一天净值就回吐全部浮盈。"""
        b = bars.get(code)
        if b is None or b.empty:
            return fallback
        v = float(b["close"].iloc[-1])
        return v if v > 0 else fallback

    # 当日所有写库动作先攒着，最后一个事务一次写完。原来是边算边写、每步各开
    # 一个连接：卖出已记成交、已删持仓，现金和 last_date 却在最后才更新——
    # 中途任何一步抛异常，卖出所得就凭空消失，而重跑会被当成新的一天再卖一遍。
    # 前向记录不可再生，宁可整天不落库，也不能落半天。
    trade_rows: list[tuple] = []
    sold: list[str] = []
    bought: list[tuple] = []

    # 到期续持的名单：与下面买入环节同一套筛选和排序，只是把已持有的也算进去。
    # 与引擎 run() 里的 renew 逐字对应（自检「逐笔等价」盯着两边）。
    renew = None
    if renew_ranked and max_hold > 0 and regime_on:
        elig = [(v["score"], c) for c, v in sig.items()
                if v["entry"] and px_at(c, as_of, "open") is not None
                and px_at(c, prev, "close") is not None]
        elig.sort(key=lambda x: -x[0] if x[0] == x[0] else 9e9)
        renew = {c for _, c in elig[:max_pos]}

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
        if expired and not s["exit"] and renew is not None and code in renew:
            h["hold_days"] = 0                       # 仍在前 N 名：续持，重新计时
            continue
        if not s["exit"] and not expired:
            continue
        if hit_limit_down(o, pc, price_limit(code, h["name"])):
            events["blocked"].append(f"{code} {h['name']} 开盘跌停，卖不出")
            continue
        gross = o * h["shares"]
        proceeds = gross - sell_cost(gross)
        cash += proceeds
        pnl = proceeds - h["cost"] * h["shares"]
        reason = (f"持有满 {max_hold} 日到期" if expired and not s["exit"]
                  else strat.reason(s["row"], "SELL"))
        trade_rows.append((account, code, h["name"], h["open_date"], as_of,
                           h["shares"], round(h["cost"], 3), round(o, 3),
                           round(pnl, 2), round(pnl / (h["cost"] * h["shares"]), 4),
                           h["hold_days"], h["open_reason"], reason))
        sold.append(code)
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
        # 按今日开盘价估值（下单时只知道开盘价），停牌的按最后收盘价，
        # 与引擎 run() 的 mv_now 逐字对应
        mv = sum(h["shares"] * (px_at(c, as_of, "open")
                                or last_close(c, h["cost"]))
                 for c, h in holds.items())
        budget = (cash + mv) / max_pos
        for _, code in cands:
            if slots <= 0:
                break
            o, pc = px_at(code, as_of, "open"), px_at(code, prev, "close")
            if o is None or pc is None:
                continue
            name = names.get(code, code)
            if hit_limit_up(o, pc, price_limit(code, name)):
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
            bought.append((account, code, name, shares, cost, as_of, reason, cost))
            holds[code] = {"name": name, "shares": shares, "cost": cost,
                           "open_date": as_of, "open_reason": reason,
                           "peak": cost, "hold_days": 0}
            events["buys"].append({"code": code, "name": name, "price": round(o, 2),
                                   "shares": shares, "amount": round(gross + fee, 2),
                                   "reason": reason})
            slots -= 1

    # ---------------- 3. 盯市 ----------------
    mv = 0.0
    updates = []
    for code, h in holds.items():
        c_px = px_at(code, as_of, "close")
        if c_px is not None:
            h["peak"] = max(h["peak"], c_px)
        mv += h["shares"] * (c_px or last_close(code, h["cost"]))
        updates.append((h["peak"], h["hold_days"], account, code))
    equity = cash + mv

    bench = None
    bd = store.load_daily(["IDX000300"], end=as_of)
    if not bd.empty:
        bench = float(bd.sort_values("date")["close"].iloc[-1])

    with store.connect() as c:                  # 一个事务：要么整天落库，要么都不落
        c.executemany("""INSERT INTO paper_trade (account,code,name,open_date,
            close_date,shares,open_price,close_price,pnl,pnl_pct,hold_days,
            open_reason,close_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      trade_rows)
        c.executemany("DELETE FROM paper_holding WHERE account=? AND code=?",
                      [(account, code) for code in sold])
        c.executemany("""INSERT INTO paper_holding
            (account,code,name,shares,cost,open_date,open_reason,peak,hold_days)
            VALUES (?,?,?,?,?,?,?,?,0)""", bought)
        c.executemany("UPDATE paper_holding SET peak=?, hold_days=? "
                      "WHERE account=? AND code=?", updates)
        c.execute("""INSERT OR REPLACE INTO paper_equity
                     (account,date,equity,cash,positions,bench,note)
                     VALUES (?,?,?,?,?,?,?)""",
                  (account, as_of, round(equity, 2), round(cash, 2), len(holds),
                   bench, "" if regime_on else "择时空仓"))
        c.executemany("INSERT INTO paper_meta (account,key,value) VALUES (?,?,?) "
                      "ON CONFLICT(account,key) DO UPDATE SET value=excluded.value",
                      [(account, "cash", str(cash)), (account, "last_date", as_of)])

    events["equity"] = round(equity, 2)
    events["cash"] = round(cash, 2)
    events["positions"] = len(holds)
    return events


def catch_up(max_days: int = 400, verbose: bool = True,
             account: str = DEFAULT_ACCOUNT) -> int:
    """从上次处理到的地方**逐日**推进到最新已收盘交易日。

    必须自己列出日期一天天走。这里曾经是循环调用 advance()（不带 as_of），
    结果只推进一天就停：advance 省略 as_of 时直接取最后一个完整交易日，
    第一次就跳到最新那天并把 last_date 设成它，第二次立刻被「已处理过」
    的守卫挡掉，循环结束。

    中间那些交易日被整段跳过，前向记录留下空洞——而 last_date 照样显示
    「已追平到最新交易日」，从外面一点都看不出来。实测踩过：账户落后 5 个
    交易日，跑完只推进了 1 天，09-02 到 09-07 全没了。
    """
    _init()
    complete = store.last_complete_day()
    if not complete:
        return 0

    last = _meta("last_date", "", account=account)
    if last:
        pending = [d for d in store.trading_days(end=complete) if d > last]
    else:
        # 新账户从建立之后才开始记，不回补历史，只处理最新那天
        pending = [complete]

    n = 0
    for d in pending[:max_days]:
        ev = advance(as_of=d, verbose=False, account=account)
        if ev.get("skipped"):
            continue
        n += 1
        if verbose and (ev["buys"] or ev["sells"]):
            print(f"  {ev['date']}  买{len(ev['buys'])} 卖{len(ev['sells'])}  "
                  f"净值 {ev['equity']:,.0f}")
    return n


def lag_days(account: str = DEFAULT_ACCOUNT) -> int:
    """账户落后于本地库最后一个完整交易日多少个交易日。

    断链是这套东西最大的失败模式——不报错，只是悄悄停在某一天，半年后才发现
    前向记录只有三天。所以它必须能被查询、被显示出来。
    """
    last = _meta("last_date", "", account=account)
    complete = store.last_complete_day()
    if not complete:
        return 0
    if not last:
        # 新建但一次都没推进过的账户：它从建立之后才开始记，不需要追历史。
        # 这里曾经返回 len(days)，于是界面上显示「模拟盘落后 2108 个交易日」，
        # 看着像坏了——而实际上 catch_up 只会推进一天（advance 在 as_of 省略时
        # 直接取最后一个完整交易日）。落后 0 天才是实情。
        return 0
    days = store.trading_days(end=complete)
    return len([d for d in days if d > last])


def data_lag_days() -> int:
    """本地行情落后真实市场约几个交易日。

    只能估：本地库自己就是交易日历，它无从知道尚未同步的那几天是不是交易日。
    按工作日数，不扣法定假日，所以长假之后会偏大——它的用途是提示「该同步了」，
    不需要精确。lag_days 答的是另一半（模拟盘落后本地库几天），两段都断得了。
    """
    from .data import source

    complete = store.last_complete_day()
    if not complete:
        return 0
    today = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))
    n = len(pd.bdate_range(pd.Timestamp(complete) + pd.Timedelta(days=1), today))
    if n and not source.market_closed_today():
        n -= 1                      # 今天还没收盘，不算落后
    return max(0, n)


def status(account: str = DEFAULT_ACCOUNT) -> dict:
    _init()
    init_cash = float(_meta("initial_cash", 0, account=account) or 0)
    with store.connect() as c:
        eq = pd.read_sql("SELECT * FROM paper_equity WHERE account=? ORDER BY date",
                         c, params=(account,))
        hold = pd.read_sql("SELECT * FROM paper_holding WHERE account=?",
                           c, params=(account,))
        tr = pd.read_sql("SELECT * FROM paper_trade WHERE account=? "
                         "ORDER BY close_date", c, params=(account,))
    out = {"account": account, "strategy": _meta("strategy", account=account),
           "params": json.loads(_meta("params", "{}", account=account)),
           "initial_cash": init_cash,
           "cash": float(_meta("cash", 0, account=account) or 0),
           "max_positions": int(_meta("max_positions", 5, account=account) or 5),
           "top": int(_meta("top", 400, account=account) or 400),
           "max_hold_days": int(_meta("max_hold_days", 0, account=account) or 0),
           "renew_ranked": bool(int(_meta("renew_ranked", 0, account=account) or 0)),
           "created_at": _meta("created_at", account=account),
           "last_date": _meta("last_date", account=account),
           "days": len(eq), "holdings": hold.to_dict("records"),
           "closed_trades": len(tr)}
    if out["strategy"]:
        out["lag_days"] = lag_days(account)
        out["data_lag_days"] = data_lag_days()
    if not eq.empty:
        e = eq["equity"].to_numpy()
        out["equity"] = float(e[-1])
        out["total_return"] = float(e[-1] / init_cash - 1) if init_cash else 0.0
        # 峰值从初始资金算起，与回测的 metrics.compute 同口径
        peak = pd.Series(e).cummax().to_numpy()
        if init_cash:
            peak = peak.clip(min=init_cash)
        out["max_drawdown"] = float(min((e / peak - 1).min(), 0.0))
        b = eq["bench"].dropna()
        if len(b) > 1:
            out["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        out["idle_days"] = int((eq["note"] == "择时空仓").sum())
        # 指数点位（4000 上下）和账户净值（20 万）差两个数量级，同图画的话
        # 净值会被压成一条平线。归一到同一起点再传，前端直接画就是对的。
        bv = eq["bench"].astype(float).ffill()
        base = float(bv.dropna().iloc[0]) if bv.notna().any() else 0.0
        out["curve"] = {
            "dates": eq["date"].tolist(),
            "equity": [round(float(x), 2) for x in eq["equity"]],
            "bench": [None if (x != x or not base) else round(init_cash * x / base, 2)
                      for x in bv.to_numpy(dtype=float)],
            "positions": [int(x) for x in eq["positions"]],
        }
    if not tr.empty:
        out["win_rate"] = float((tr["pnl"] > 0).mean())
        out["realized_pnl"] = float(tr["pnl"].sum())
    return out


def decay_report(account: str = DEFAULT_ACCOUNT, min_days: int = 20,
                 years: int = 4, refresh: bool = False) -> dict:
    """前向表现落在该策略历史同长度窗口分布的第几分位。

    **别把这读成「模拟盘 vs 回测」的对账。** 自检里那项「模拟盘与回测引擎逐笔等价」
    已经证明两边在同一区间上完全一致，拿它们相减恒为零——做出来是个好看的、
    零信息量的仪表盘。引擎失真该由自检去管。

    这里答的是另一个问题：**这个策略是不是已经死了。** 把前向那段收益放回它自己
    历史上所有等长窗口的分布里，看它排第几。同时给出同期基准的分位——如果策略
    在低分位而大盘也在低分位，那是市场环境，不是策略衰减，这两件事必须能分开。

    单看一次分位数说明不了什么（样本短、噪声占绝对主导），它的用法是长期盯着：
    分位持续贴地，才值得怀疑。
    """
    import numpy as np

    from .backtest import engine

    st = status(account)
    if not st.get("strategy"):
        return {"error": f"账户 {account} 不存在"}

    days = st["days"]
    if days < min_days:
        return {"enough": False, "days": days, "min_days": min_days,
                "note": f"前向只有 {days} 个交易日，不足 {min_days} 个。"
                        "样本太短的分位数是纯噪声，不给结论。"}

    key = f"decay_{days}"
    if not refresh:
        cached = _meta(key, account=account)
        if cached:
            return json.loads(cached)

    fwd = st["total_return"]
    first = st["curve"]["dates"][0]
    prior = store.trading_days(end=first)
    if len(prior) < days + 260:
        return {"enough": False, "days": days,
                "note": f"模拟盘起点 {first} 之前只有 {len(prior)} 个交易日，"
                        "历史不足以构造分布。"}

    hist_end = prior[-2]                      # 严格早于模拟盘起点
    hist_start = (pd.Timestamp(hist_end)
                  - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    uni = universe.build(as_of=hist_start)
    if uni.empty:
        return {"enough": False, "note": f"{hist_start} 当日无法构建股票池"}
    codes = uni["code"].tolist()[:st["top"]]
    names = dict(zip(uni["code"], uni["name"]))

    strat = get_strategy(st["strategy"], **st["params"])
    cfg = engine.BacktestConfig(initial_cash=st["initial_cash"],
                                max_positions=st["max_positions"],
                                max_hold_days=st["max_hold_days"],
                                renew_ranked=st["renew_ranked"])
    r = engine.run(strat, codes, hist_start, hist_end, cfg, names)
    if r.equity.empty:
        return {"enough": False,
                "note": r.metrics.get("error", "历史回测没有结果")}

    v = r.equity["equity"].to_numpy(dtype=float)
    if len(v) <= days:
        return {"enough": False, "note": "历史净值序列短于前向窗口"}
    wins = v[days:] / v[:-days] - 1

    # 同期基准的分位：策略低分位 + 大盘也低分位 = 市场问题，不是策略问题
    bench_pct = None
    fwd_bench = st.get("benchmark_return")
    if fwd_bench is not None and "benchmark" in r.equity:
        b = r.equity["benchmark"].ffill().to_numpy(dtype=float)
        if np.isfinite(b).all() and len(b) > days:
            bwins = b[days:] / b[:-days] - 1
            bench_pct = float((bwins < fwd_bench).mean())

    out = {
        "enough": True, "days": days,
        "forward_return": fwd, "forward_bench": fwd_bench,
        "percentile": float((wins < fwd).mean()),
        "bench_percentile": bench_pct,
        "windows": int(len(wins)),
        "hist_median": float(np.median(wins)),
        "hist_p10": float(np.percentile(wins, 10)),
        "hist_p90": float(np.percentile(wins, 90)),
        "hist_start": hist_start, "hist_end": hist_end,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 缓存键带天数，不清的话每过一天就多攒一条，一年 250 条全是过期的
    with store.connect() as c:
        c.execute("DELETE FROM paper_meta WHERE account=? "
                  "AND key LIKE 'decay%'", (account,))
    _set(key, json.dumps(out), account=account)
    return out


def trades(account: str = DEFAULT_ACCOUNT, limit: int = 200) -> list[dict]:
    _init()
    with store.connect() as c:
        df = pd.read_sql("SELECT * FROM paper_trade WHERE account=? "
                         "ORDER BY close_date DESC, id DESC LIMIT ?",
                         c, params=(account, limit))
    return df.to_dict("records")
