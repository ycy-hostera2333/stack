"""本地行情库：SQLite 存储 + 增量更新。

设计取舍：全市场 5000+ 只股票 × 多年日线约千万级行，SQLite 单文件足够扛，
且免去额外服务依赖。所有查询都走 (code, date) 主键索引，按需加载而非全量入内存。
"""
from __future__ import annotations

import bisect
import os
import sqlite3
import threading
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    code        TEXT PRIMARY KEY,
    name        TEXT,
    board       TEXT,
    is_st       INTEGER DEFAULT 0,
    listed_date TEXT,      -- 上市日期，用于剔除次新股
    delisted_date TEXT,    -- 退市日期；为空表示仍在市
    status      TEXT DEFAULT 'listed',   -- listed / delisted
    industry    TEXT,      -- 所属行业（深市官方提供，沪市可能为空）
    float_share REAL,      -- 流通股本（股）
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS daily (
    code     TEXT NOT NULL,
    date     TEXT NOT NULL,   -- YYYY-MM-DD
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,            -- 手
    amount   REAL,            -- 元
    pct_chg  REAL,            -- 涨跌幅 %
    turnover REAL,            -- 换手率 %
    PRIMARY KEY (code, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    code      TEXT NOT NULL,
    name      TEXT,
    shares    INTEGER NOT NULL,
    cost      REAL NOT NULL,     -- 每股成本价
    open_date TEXT,
    note      TEXT
);

CREATE TABLE IF NOT EXISTS signal_log (
    date     TEXT NOT NULL,
    strategy TEXT NOT NULL,
    code     TEXT NOT NULL,
    name     TEXT,
    action   TEXT NOT NULL,      -- BUY / SELL
    price    REAL,
    reason   TEXT,
    PRIMARY KEY (date, strategy, code, action)
);

CREATE TABLE IF NOT EXISTS simulator_saves (
    user_name  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_errors (
    code      TEXT NOT NULL,
    date      TEXT NOT NULL,
    reason    TEXT,
    attempt   INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (code, date)
);

-- 网页上写的自定义策略。代码本身存在这里，服务启动时动态编译进策略注册表，
-- 与内置策略走完全相同的路径（回测/信号/模拟盘/命令行都不区分）。
-- name 既是主键，也是注册名：不能和内置策略重名，否则内置策略会被静默顶掉。
CREATE TABLE IF NOT EXISTS user_strategies (
    name        TEXT PRIMARY KEY,
    label       TEXT,
    description TEXT,
    code        TEXT NOT NULL,
    created_at  TEXT,
    updated_at  TEXT
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ 读缓存
# 回测/信号/模拟盘读的都是同一份行情，而读一次全市场是整个流程里最慢的一步
# （150 万行约 4-8 秒，大头在 sqlite 逐行转 Python 对象，换表结构也省不下来）。
# 网页上改个参数重跑回测，股票池和区间都没变，却要把同样的数据再读一遍。
#
# 失效判据用 SQLite 自己的 PRAGMA data_version：在一条常驻连接上反复查询，
# 只要**任何其他连接**（包括别的进程，比如另开的命令行同步）提交过写入，
# 返回值就会变。比文件修改时间可靠——WAL 文件在检查点之后是原地覆写的，
# 大小不变，修改时间在一些文件系统上又只有毫秒级精度。
#
# data_version 的数值只在**同一条连接**内有意义：换一条新连接它会从头数起，
# 可能恰好数回某条旧缓存的标签。所以版本号是 (代次, data_version)，每开一条
# 新的监视连接代次加一，新旧连接的数永远不会被拿来比较。
# 库文件被整个换掉（恢复备份、拷来别的机器上的库）时，POSIX 上旧连接还开着旧文件，
# data_version 永远不变也不报错——所以每次还要比一下文件身份 (st_dev, st_ino)。
# （Windows 上这条常驻连接会占着 market.db，服务运行期间删不掉、换不了它。）
_watch_lock = threading.Lock()
_watch: dict = {"conn": None, "gen": 0, "ident": None}


def _file_ident():
    try:
        st = os.stat(DB_PATH)
        return (st.st_dev, st.st_ino)
    except OSError:
        return None


def _drop_watch() -> None:
    c = _watch["conn"]
    _watch["conn"] = None
    if c is not None:
        try:
            c.close()
        except sqlite3.Error:
            pass


def data_version() -> tuple | None:
    """库内容的版本号：任何连接提交写入后都会变。只用来判断缓存是否过期。

    返回 None 表示这次判断不了，调用方不能用缓存、也不该写缓存。
    """
    with _watch_lock:
        if _watch["conn"] is not None and _file_ident() != _watch["ident"]:
            _drop_watch()                    # 文件被换掉了
        if _watch["conn"] is None:
            try:
                c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
            except sqlite3.Error:
                return None
            _watch.update(conn=c, gen=_watch["gen"] + 1, ident=_file_ident())
        try:
            dv = int(_watch["conn"].execute("PRAGMA data_version").fetchone()[0])
        except sqlite3.Error:
            _drop_watch()
            return None
        return (_watch["gen"], dv)


_CACHE_MAX = 2                 # 一份全市场日线约 100+MB，只留最近两份
_CACHE_MIN_CODES = 100         # 小查询（单只、指数、几十只）本来就快，不缓存
_daily_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()
_days_cache: dict = {}


def clear_cache() -> None:
    with _cache_lock:
        _daily_cache.clear()
        _days_cache.clear()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        # 已有库的平滑升级：老版本的 instruments 表没有退市字段
        have = {r[1] for r in conn.execute("PRAGMA table_info(instruments)")}
        for col, ddl in (("delisted_date", "TEXT"),
                         ("status", "TEXT DEFAULT 'listed'")):
            if col not in have:
                conn.execute(f"ALTER TABLE instruments ADD COLUMN {col} {ddl}")


# ------------------------------------------------------------------ 写入
def _rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame → sqlite 可绑定的元组列表。

    sqlite3 不认识 pandas 的 NA/NaT 和 numpy 标量类型，统一转成 None 和原生 int/float。
    """
    out = []
    for row in df.itertuples(index=False, name=None):
        vals = []
        for v in row:
            if v is None or (isinstance(v, float) and v != v):
                vals.append(None)
            elif v is pd.NA or v is pd.NaT:
                vals.append(None)
            elif isinstance(v, (np.integer,)):
                vals.append(int(v))
            elif isinstance(v, (np.floating,)):
                f = float(v)
                vals.append(None if f != f else f)
            elif isinstance(v, np.bool_):
                vals.append(int(v))
            elif isinstance(v, str):
                vals.append(v)
            else:
                try:
                    vals.append(None if pd.isna(v) else v)
                except (TypeError, ValueError):
                    vals.append(v)
        out.append(tuple(vals))
    return out


def upsert_instruments(df: pd.DataFrame) -> int:
    """写入/更新股票基础信息。"""
    if df.empty:
        return 0
    cols = ["code", "name", "board", "is_st", "listed_date", "delisted_date",
            "status", "industry", "float_share", "updated_at"]
    df = df.reindex(columns=cols)
    with connect() as conn:
        conn.executemany(
            f"INSERT INTO instruments ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
            "ON CONFLICT(code) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in cols[1:]),
            _rows(df),
        )
    return len(df)


# 新数据里为空时保留库里原值的列。腾讯/BaoStock 的 pct_chg 是由收盘价现算的，
# 抓回来那一段的第一行必然是空；增量同步每次往回重抓 7 天、整行覆盖，于是每同步
# 一次，重叠窗口第一天原本好好的涨跌幅就被写成空，日积月累。
_KEEP_IF_NULL = ("amount", "pct_chg", "turnover")


def upsert_daily(df: pd.DataFrame) -> int:
    """写入日线。重复的 (code,date) 覆盖，便于修正复权后的历史价格。

    价格、成交量整行以新数据为准；_KEEP_IF_NULL 里的列新数据为空时保留原值。
    """
    if df is None or df.empty:
        return 0
    cols = ["code", "date", "open", "high", "low", "close",
            "volume", "amount", "pct_chg", "turnover"]
    df = df.reindex(columns=cols)
    sets = ", ".join(
        f"{c}=COALESCE(excluded.{c}, daily.{c})" if c in _KEEP_IF_NULL
        else f"{c}=excluded.{c}" for c in cols[2:])
    with connect() as conn:
        conn.executemany(
            f"INSERT INTO daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))}) "
            f"ON CONFLICT(code, date) DO UPDATE SET {sets}",
            _rows(df),
        )
    return len(df)


def repair_pct_chg(codes: Sequence[str] | None = None) -> int:
    """把 pct_chg 为空、前一根收盘价已知的行补上（前复权收盘价的日涨幅）。

    补的是增量同步留下的空洞（见 _KEEP_IF_NULL）。逐只股票各开一个事务：
    一条 UPDATE 扫全表会长时间占着写锁，模拟盘推进等 30 秒就会报「database is locked」。
    每只股票上市第一天本来就没有前收，留空。返回补上的行数。
    """
    with connect() as conn:
        if codes is None:
            codes = [r[0] for r in conn.execute(
                "SELECT DISTINCT code FROM daily WHERE pct_chg IS NULL "
                "AND code NOT LIKE 'IDX%'")]
    n = 0
    for code in codes:
        with connect() as conn:
            cur = conn.execute("""
                UPDATE daily SET pct_chg = (
                    SELECT (daily.close / p.close - 1) * 100 FROM daily AS p
                    WHERE p.code = daily.code AND p.date < daily.date
                    ORDER BY p.date DESC LIMIT 1)
                WHERE code = ? AND pct_chg IS NULL AND close > 0
                  AND (SELECT p.close FROM daily AS p
                       WHERE p.code = daily.code AND p.date < daily.date
                       ORDER BY p.date DESC LIMIT 1) > 0""", (code,))
            n += max(cur.rowcount, 0)
    return n


def replace_daily(code: str, df: pd.DataFrame) -> int:
    """整只替换：先删该股全部日线，再写入 df。同一个事务，中途失败不会留下半截。

    除权后重建历史必须用这个而不是 upsert：前复权会把整段历史一起改掉，
    upsert 只覆盖新数据里有的日子，新数据里恰好缺的那几天会留着旧口径的价格。
    """
    if df is None or df.empty:
        return 0
    cols = ["code", "date", "open", "high", "low", "close",
            "volume", "amount", "pct_chg", "turnover"]
    df = df.reindex(columns=cols)
    with connect() as conn:
        conn.execute("DELETE FROM daily WHERE code=?", (code,))
        conn.executemany(
            f"INSERT INTO daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            _rows(df),
        )
    return len(df)


def set_meta(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO meta (key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_meta(key: str, default: str | None = None) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


# ------------------------------------------------------------------ 读取
def load_instruments() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql("SELECT * FROM instruments", conn)


def instruments_with_data(query: str = "", limit: int = 80) -> pd.DataFrame:
    """按代码或名称搜索已有日线的标的，避免把整张 daily 表读入内存。"""
    sql = ("SELECT i.code, i.name FROM instruments i "
           "WHERE EXISTS (SELECT 1 FROM daily d WHERE d.code=i.code)")
    params: list = []
    if query.strip():
        sql += " AND (i.code LIKE ? OR i.name LIKE ?)"
        term = f"%{query.strip()}%"
        params.extend([term, term])
    sql += " ORDER BY i.code LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def save_simulator(user_name: str, payload: str, updated_at: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO simulator_saves (user_name,payload,updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_name) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
            (user_name, payload, updated_at),
        )


def load_simulator(user_name: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT payload,updated_at FROM simulator_saves WHERE user_name=?", (user_name,)).fetchone()
    if not row:
        return None
    return {"payload": row[0], "updated_at": row[1]}


def list_simulator_saves() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT user_name,updated_at FROM simulator_saves ORDER BY updated_at DESC").fetchall()
    return [{"user_name": r[0], "updated_at": r[1]} for r in rows]


def last_dates() -> dict[str, str]:
    """每只股票本地已有的最新日期，用于增量更新。"""
    with connect() as conn:
        rows = conn.execute("SELECT code, MAX(date) FROM daily GROUP BY code").fetchall()
    return dict(rows)


def load_daily(
    codes: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """加载日线，返回 long 格式 DataFrame（code, date, ohlcv...），date 为 datetime。

    注：曾试过在代码数量多时改走日期索引再用 pandas 过滤，实测反而慢 4 倍
    （2438 只 / 600 天：IN 子句 3.2s，日期索引 13.6s），因为后者要先读出全市场的行。
    (code,date) 主键在这里已经够用，维持 IN 子句。
    """
    if codes is not None:
        codes = list(codes)
        if not codes:
            return pd.DataFrame()
    big = codes is None or len(codes) >= _CACHE_MIN_CODES
    if big:
        key = (tuple(codes) if codes is not None else None, start, end)
        ver = data_version()
        with _cache_lock:
            hit = _daily_cache.get(key)
            if hit is not None and ver is not None and hit[0] == ver:
                _daily_cache.move_to_end(key)
                # 给副本：调用方可以随意改列（api 的回放接口就会改 date 列）
                return hit[1].copy()
    df = _load_daily_sql(codes, start, end)
    if big and ver is not None:
        with _cache_lock:
            _daily_cache[key] = (ver, df)
            _daily_cache.move_to_end(key)
            while len(_daily_cache) > _CACHE_MAX:
                _daily_cache.popitem(last=False)
        return df.copy()
    return df


def _load_daily_sql(codes: list[str] | None, start: str | None,
                    end: str | None) -> pd.DataFrame:
    sql = "SELECT * FROM daily WHERE 1=1"
    params: list = []
    if codes is not None:
        sql += f" AND code IN ({','.join('?' * len(codes))})"
        params += codes
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY code, date"
    with connect() as conn:
        df = pd.read_sql(sql, conn, params=params)
    if not df.empty:
        # 显式给格式：不给的话 pandas 要逐个推断，全市场一次慢 2-3 倍
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def usable_history(g: pd.DataFrame) -> pd.DataFrame:
    """截掉前复权价 <= 0 的历史前缀，只保留价格转正之后的部分。

    前复权是从历史价里扣减累计分红，当累计分红超过当年股价时，复权价会变成负数。
    这不是数据错误，是前复权本身的性质——但在负价格上算出来的 MA/RSI/动量/波动率
    全无意义，而且跨越正负分界的窗口会把垃圾传染给之后的正常区间。

    实测本地库里有 23 只股票存在这种情况，最严重的 601919 有 714/2077 根是负的。

    g 需按日期升序。返回最后一根非正价 K 线之后的所有数据。
    """
    if g.empty:
        return g
    # 走 numpy：逐只调用，三次 pandas 比较 + 或运算的固定开销比计算本身大得多
    bad = ((g["close"].to_numpy() <= 0) | (g["low"].to_numpy() <= 0)
           | (g["open"].to_numpy() <= 0))
    if not bad.any():
        return g
    last_bad = np.flatnonzero(bad)[-1]
    return g.iloc[last_bad + 1:]


def iter_stocks(raw: pd.DataFrame):
    """逐只产出 (代码, 按日期升序且截掉负价前缀的日线)。

    引擎、模拟盘、每日信号、因子面板原来各写一遍
    `for code, g in raw.groupby("code"): usable_history(g.sort_values("date"))`。
    load_daily 返回的本来就按 (code, date) 排好序，这里直接按代码边界切片，
    省掉 groupby 和每只一次的排序；万一传进来的没排序，退回原来的做法。
    每只给的是副本：策略的 prepare() 原地改列不会写回调用方的大表。
    """
    if raw is None or raw.empty:
        return
    codes = raw["code"].to_numpy()
    if not pd.Index(codes).is_monotonic_increasing:
        for code, g in raw.groupby("code", sort=False):
            yield code, usable_history(g.sort_values("date"))
        return
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    for a, b in zip(starts, ends):
        g = raw.iloc[a:b]
        if not g["date"].is_monotonic_increasing:
            g = g.sort_values("date")
        yield codes[a], usable_history(g.copy())


# 残日判据：当日条数不低于近 30 个交易日中位数的这个比例。
#
# 曾经是 0.5，太松：2026-09-22 同步跑了一半，当日只有 2384/4564 行（52%），
# 照样被判成「完整」。每日信号在 3540 只的股票池里把 1675 只当成停牌静默剔除，
# 候选是从半个市场里选出来的，全程不报错。正常交易日的条数只差几十只
# （停复牌、新股），0.9 足以区分「正常波动」和「同步没跑完」。
COMPLETE_RATIO = 0.9


def pick_complete_day(rows: list[tuple[str, int]],
                      min_ratio: float = COMPLETE_RATIO) -> str | None:
    """从 [(日期, 条数), …]（日期降序）里挑最后一个完整交易日。纯函数，便于自检构造数据。"""
    if not rows:
        return None
    counts = sorted(n for _, n in rows)
    med = counts[len(counts) // 2]
    for d, n in rows:                     # rows 已按日期降序
        if n >= med * min_ratio:
            return d
    return None


def last_complete_day(min_ratio: float = COMPLETE_RATIO,
                      lookback: int = 30) -> str | None:
    """最后一个**数据完整**的交易日。

    中断的同步会在库尾留下只有几只股票的残日。它照样出现在 trading_days() 末尾，
    于是「最后一个交易日」就变成那一天，其余几千只全被判成当日停牌——
    每日信号的候选从几百只缩成几只、模拟盘会照着这份残缺名单建仓，全程不报错。
    实测 2026-08-12 库里只有 6 行，而前一交易日是 4577 行。

    注意这与 signals 里那个「跳过未收盘的当日 K 线」是两回事：那一条只防今天，
    这一条防的是任何时候留下的残日，哪怕它已经是几天前的事。

    判据见 COMPLETE_RATIO。
    """
    key = ("complete", min_ratio, lookback)
    ver = data_version()
    with _cache_lock:
        hit = _days_cache.get(key)
        if hit is not None and ver is not None and hit[0] == ver:
            return hit[1]
    with connect() as conn:
        rows = conn.execute(
            "SELECT date, COUNT(*) FROM daily GROUP BY date "
            "ORDER BY date DESC LIMIT ?", (lookback,)).fetchall()
    out = pick_complete_day(rows, min_ratio)
    with _cache_lock:
        _days_cache[key] = (ver, out)
    return out


def trading_days(start: str | None = None, end: str | None = None) -> list[str]:
    """本地库里出现过的所有交易日。

    全表 DISTINCT 要扫一遍日期索引（真实库上千万行）。模拟盘逐日推进时每推一天
    要调好几次，所以整份日历按库版本缓存一份，区间用二分切片。
    """
    ver = data_version()
    with _cache_lock:
        hit = _days_cache.get("days")
    if hit is None or ver is None or hit[0] != ver:
        with connect() as conn:
            days = [r[0] for r in conn.execute(
                "SELECT DISTINCT date FROM daily ORDER BY date").fetchall()]
        with _cache_lock:
            _days_cache["days"] = (ver, days)
    else:
        days = hit[1]
    lo = bisect.bisect_left(days, start) if start else 0
    hi = bisect.bisect_right(days, end) if end else len(days)
    return days[lo:hi]


def coverage() -> dict:
    """本地库统计，用于界面上显示数据健康度。

    指数以 IDX 前缀和个股同表存放，统计"有行情的股票数"时必须排除，
    否则会出现 4593/4589 这种分子大于分母的怪数字。
    """
    with connect() as conn:
        n_inst = conn.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date) FROM daily"
        ).fetchone()
        n_stock = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM daily WHERE code NOT LIKE 'IDX%'"
        ).fetchone()[0]
        n_index = conn.execute(
            "SELECT COUNT(DISTINCT code) FROM daily WHERE code LIKE 'IDX%'"
        ).fetchone()[0]
    return {
        "instruments": n_inst,
        "bars": row[0] or 0,
        "codes_with_data": n_stock or 0,
        "indices": n_index or 0,
        "first_date": row[2],
        "last_date": row[3],
    }


# ------------------------------------------------------------------ 持仓
def list_positions() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql("SELECT * FROM positions ORDER BY open_date DESC", conn)


def add_position(code: str, name: str, shares: int, cost: float,
                 open_date: str, note: str = "") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO positions (code,name,shares,cost,open_date,note) "
            "VALUES (?,?,?,?,?,?)",
            (code, name, shares, cost, open_date, note),
        )
        return cur.lastrowid


def delete_position(pos_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM positions WHERE id=?", (pos_id,))


# ------------------------------------------------------------------ 信号留痕
def log_signals(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO signal_log (date,strategy,code,name,action,price,reason) "
            "VALUES (:date,:strategy,:code,:name,:action,:price,:reason)",
            rows,
        )
    return len(rows)


def load_signal_log(limit: int = 200) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql(
            "SELECT * FROM signal_log ORDER BY date DESC, strategy, action LIMIT ?",
            conn, params=[limit],
        )


# ------------------------------------------------------------------ 自定义策略
def list_user_strategies(with_code: bool = False) -> pd.DataFrame:
    """网页上保存的策略列表。

    默认不取代码：列表页只需要名字和行数，把每个策略的源码全读出来没有意义。
    """
    cols = "name, label, description, created_at, updated_at"
    if with_code:
        cols += ", code"
    with connect() as conn:
        return pd.read_sql(
            f"SELECT {cols} FROM user_strategies ORDER BY updated_at DESC", conn)


def load_user_strategy(name: str) -> dict | None:
    """单个策略的完整记录（含代码）。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT name, label, description, code, created_at, updated_at "
            "FROM user_strategies WHERE name=?", (name,)).fetchone()
    if not row:
        return None
    keys = ("name", "label", "description", "code", "created_at", "updated_at")
    return dict(zip(keys, row))


def upsert_user_strategy(name: str, label: str, description: str,
                         code: str) -> None:
    """新增或覆盖保存。created_at 只在首次写入时落下，其余情况保持不变——
    界面上「建于/改于」两列要能分开看。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        conn.execute(
            "INSERT INTO user_strategies "
            "(name,label,description,code,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "label=excluded.label, description=excluded.description, "
            "code=excluded.code, updated_at=excluded.updated_at",
            (name, label, description, code, now, now))


def delete_user_strategy(name: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM user_strategies WHERE name=?", (name,))
        return cur.rowcount > 0


# ------------------------------------------------------------------ 同步失败留痕
def log_sync_error(code: str, message: str, attempt: int = 0) -> None:
    """记录一次同步失败。同一股票同一天重复失败时覆盖，并累加尝试次数。

    date 必须是**日期**而不是精确到秒的时间戳：主键是 (code, date)，
    用时间戳的话每次失败都是新行，ON CONFLICT 永不触发，表会无限增长。
    """
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO sync_errors (code,date,reason,attempt,created_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(code,date) DO UPDATE SET "
                "reason=excluded.reason, "
                "attempt=sync_errors.attempt+1, "
                "created_at=excluded.created_at",
                (code, day, message, max(attempt, 1), ts),
            )
    except sqlite3.OperationalError:
        # 老库还没有这张表：建好再写一次。留痕失败不应该中断同步本身。
        init_db()
        try:
            with connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO sync_errors "
                    "(code,date,reason,attempt,created_at) VALUES (?,?,?,?,?)",
                    (code, day, message, max(attempt, 1), ts),
                )
        except sqlite3.Error:
            pass


def load_sync_errors(limit: int = 200) -> pd.DataFrame:
    """读取最近的同步失败记录。"""
    with connect() as conn:
        return pd.read_sql(
            "SELECT * FROM sync_errors ORDER BY created_at DESC, code LIMIT ?",
            conn, params=[limit],
        )


def clear_sync_errors(codes: Sequence[str] | None = None) -> int:
    """清除失败记录。codes 为 None 时清空全部；否则只清指定股票。"""
    with connect() as conn:
        if codes is None:
            cur = conn.execute("DELETE FROM sync_errors")
        else:
            codes = list(codes)
            if not codes:
                return 0
            cur = conn.execute(
                f"DELETE FROM sync_errors WHERE code IN ({','.join('?' * len(codes))})",
                codes,
            )
    return cur.rowcount


# ------------------------------------------------------------------ 缺失检查
def find_gaps() -> dict:
    """数据缺失检查。

    找出两类问题：
    1. latest_lag：有数据但最新日期落后于全市场最新交易日的股票（滞后天数）
    2. no_data：已登记但没有日线数据的股票

    返回 {"market_last_date": str, "latest_lag": [...], "no_data": [...]}。
    """
    with connect() as conn:
        # 全市场最新交易日（排除指数 IDX 前缀）
        row = conn.execute(
            "SELECT MAX(date) FROM daily WHERE code NOT LIKE 'IDX%'"
        ).fetchone()
        market_last = row[0] if row else None

        # 每只股票的最新日期
        lag_rows = conn.execute(
            "SELECT code, MAX(date) AS last FROM daily "
            "WHERE code NOT LIKE 'IDX%' GROUP BY code"
        ).fetchall()

        # 已登记但没有日线的股票
        no_data = [r[0] for r in conn.execute(
            "SELECT code FROM instruments WHERE code NOT LIKE 'IDX%' "
            "AND NOT EXISTS (SELECT 1 FROM daily d WHERE d.code=instruments.code)"
        ).fetchall()]

        # 同步失败记录（从 sync_errors 表读取）
        err_rows = conn.execute(
            "SELECT code, reason, created_at FROM sync_errors ORDER BY created_at DESC"
        ).fetchall()

    latest_lag = []
    if market_last:
        from datetime import datetime as _dt
        m_last = _dt.strptime(market_last, "%Y-%m-%d")
        for code, last in lag_rows:
            if not last:
                continue
            lag_days = (m_last - _dt.strptime(last, "%Y-%m-%d")).days
            if lag_days > 3:   # 交易日的自然日约 3 天（含周末）
                latest_lag.append({
                    "code": code,
                    "last_date": last,
                    "lag_days": lag_days,
                })
        latest_lag.sort(key=lambda x: x["lag_days"], reverse=True)

    failed = [{"code": r[0], "reason": r[1], "created_at": r[2]} for r in err_rows]

    return {
        "market_last_date": market_last,
        "latest_lag": latest_lag,
        "no_data": no_data,
        "failed": failed,
    }
