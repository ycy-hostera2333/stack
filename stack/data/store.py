"""本地行情库：SQLite 存储 + 增量更新。

设计取舍：全市场 5000+ 只股票 × 多年日线约千万级行，SQLite 单文件足够扛，
且免去额外服务依赖。所有查询都走 (code, date) 主键索引，按需加载而非全量入内存。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
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


def upsert_daily(df: pd.DataFrame) -> int:
    """写入日线。重复的 (code,date) 覆盖，便于修正复权后的历史价格。"""
    if df is None or df.empty:
        return 0
    cols = ["code", "date", "open", "high", "low", "close",
            "volume", "amount", "pct_chg", "turnover"]
    df = df.reindex(columns=cols)
    with connect() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO daily ({','.join(cols)}) "
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
    sql = "SELECT * FROM daily WHERE 1=1"
    params: list = []
    if codes is not None:
        codes = list(codes)
        if not codes:
            return pd.DataFrame()
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
        df["date"] = pd.to_datetime(df["date"])
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
    bad = (g["close"] <= 0) | (g["low"] <= 0) | (g["open"] <= 0)
    if not bad.any():
        return g
    last_bad = np.flatnonzero(bad.to_numpy())[-1]
    return g.iloc[last_bad + 1:]


def trading_days(start: str | None = None, end: str | None = None) -> list[str]:
    """本地库里出现过的所有交易日。"""
    sql = "SELECT DISTINCT date FROM daily WHERE 1=1"
    params: list = []
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    sql += " ORDER BY date"
    with connect() as conn:
        return [r[0] for r in conn.execute(sql, params).fetchall()]


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
