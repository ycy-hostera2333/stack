"""基本面数据：抓取、point-in-time 存储与查询。

## 为什么必须做 point-in-time

财报是季频且**有公告滞后**：2024 年报要到 2025 年 4 月底才披露完。
如果按报告期（2024-12-31）对齐，回测在 2025 年 1 月就用上了 4 月才知道的数据——
这是最隐蔽的一类前视偏差，因为代码看起来完全合理，结果也不会报错，
只会让所有基本面策略的回测凭空变好。

## 可用日期怎么定

理想是用**首次公告日**。实测 akshare 的 `stock_yjbb_em` 虽然带「最新公告日期」列，
但那是该公司**最近一次任何公告**的日期，不是该期报告的首次公告日
（例：茅台 2023 年报那行显示 2025-04-03，而 2023 年报实际 2024 年 4 月就公告了）。
所以该列不能用作 PIT 锚点。

改用**法定披露截止日**：

    Q1  (03-31) -> 当年 04-30
    中报(06-30) -> 当年 08-31
    Q3  (09-30) -> 当年 10-31
    年报(12-31) -> **次年** 04-30

这是保守做法：**永远不会提前使用未公开的数据**。代价是提前披露的公司
要等到截止日才被用上，损失一点时效性，换取可证明的零前视偏差。

要更精确的公告日需要 Tushare 的 fina_indicator（带真实 ann_date），
配置 token 后可自行扩展。
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import store

# 报告期 -> 法定披露截止日（月, 日, 跨年偏移）
DEADLINE = {"0331": (4, 30, 0), "0630": (8, 31, 0),
            "0930": (10, 31, 0), "1231": (4, 30, 1)}

_COLS = {
    "每股收益": "eps",
    "每股净资产": "bps",
    "净资产收益率": "roe",
    "销售毛利率": "gross_margin",
    "每股经营现金流量": "ocfps",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "净利润-净利润": "profit",
    "净利润-同比增长": "profit_yoy",
    "所处行业": "industry",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    code       TEXT NOT NULL,
    period     TEXT NOT NULL,   -- 报告期 YYYYMMDD
    avail_date TEXT NOT NULL,   -- 可用日期（法定披露截止日），PIT 的关键
    eps REAL, bps REAL, roe REAL, gross_margin REAL, ocfps REAL,
    revenue REAL, revenue_yoy REAL, profit REAL, profit_yoy REAL,
    industry TEXT,
    PRIMARY KEY (code, period)
);
CREATE INDEX IF NOT EXISTS idx_fund_avail ON fundamentals(avail_date);
"""


def init() -> None:
    with store.connect() as c:
        c.executescript(SCHEMA)


def avail_date(period: str) -> str:
    """报告期 -> 法定披露截止日。"""
    y, md = int(period[:4]), period[4:]
    m, d, off = DEADLINE[md]
    return f"{y + off:04d}-{m:02d}-{d:02d}"


def periods(start_year: int = 2017, end: str | None = None) -> list[str]:
    """列出所有已过披露截止日的报告期。

    未过截止日的报告期不纳入——那部分数据现在还不该被任何回测看到。
    """
    today = end or datetime.now().strftime("%Y-%m-%d")
    out = []
    for y in range(start_year, int(today[:4]) + 1):
        for md in ("0331", "0630", "0930", "1231"):
            p = f"{y}{md}"
            if avail_date(p) <= today:
                out.append(p)
    return out


def fetch_period(period: str) -> pd.DataFrame:
    """抓取单个报告期的全市场业绩数据。一次请求覆盖全部股票。"""
    import akshare as ak
    raw = ak.stock_yjbb_em(date=period)
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame({"code": raw["股票代码"].astype(str).str.zfill(6)})
    for src, dst in _COLS.items():
        out[dst] = raw[src] if src in raw.columns else pd.NA
    for c in ("eps", "bps", "roe", "gross_margin", "ocfps",
              "revenue", "revenue_yoy", "profit", "profit_yoy"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["period"] = period
    out["avail_date"] = avail_date(period)
    return out.dropna(subset=["code"]).drop_duplicates(subset=["code"])


def sync(start_year: int = 2017, progress=None, sleep: float = 0.8) -> dict:
    """同步所有报告期。约 34 个季度，每期一次请求。"""
    init()
    ps = periods(start_year)
    stats = {"periods": len(ps), "ok": 0, "failed": 0, "rows": 0}
    cols = ["code", "period", "avail_date", "eps", "bps", "roe", "gross_margin",
            "ocfps", "revenue", "revenue_yoy", "profit", "profit_yoy", "industry"]
    for i, p in enumerate(ps, 1):
        try:
            df = fetch_period(p)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            stats["failed"] += 1
        else:
            df = df.reindex(columns=cols)
            with store.connect() as c:
                c.executemany(
                    f"INSERT OR REPLACE INTO fundamentals ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", store._rows(df))
            stats["ok"] += 1
            stats["rows"] += len(df)
        if progress:
            progress(i, len(ps), stats)
        time.sleep(sleep)
    return stats


def load_pit(codes: list[str] | None = None,
             start: str | None = None) -> pd.DataFrame:
    """读取基本面数据（long 格式，含 avail_date）。"""
    sql = "SELECT * FROM fundamentals WHERE 1=1"
    params: list = []
    if codes:
        sql += f" AND code IN ({','.join('?' * len(codes))})"
        params += list(codes)
    if start:
        sql += " AND avail_date >= ?"
        params.append(start)
    sql += " ORDER BY code, avail_date"
    with store.connect() as c:
        return pd.read_sql(sql, c, params=params)


def as_panel(field: str, dates: list[str],
             codes: list[str]) -> pd.DataFrame:
    """把某个基本面字段展开成 point-in-time 宽表（日期 × 股票）。

    每个交易日取**该日之前已过披露截止日**的最新一期数据，用 ffill 实现。
    这样 2025-01-15 这天看到的仍是 2024 年三季报，直到 2025-04-30 才切到年报——
    与真实世界的信息可得性一致。
    """
    fd = load_pit(codes)
    if fd.empty or field not in fd.columns:
        return pd.DataFrame(index=pd.Index(dates, name="date"), columns=codes,
                            dtype="float32")
    fd = fd[["code", "period", "avail_date", field]].dropna(subset=[field])
    # 年报(1231)与次年一季报(0331)的法定截止日**都是 4-30**，会撞在同一天。
    # 撞车时必须取更新的那期（一季报），所以先按 period 升序，再用 last。
    fd = fd.sort_values(["code", "avail_date", "period"])
    wide = (fd.pivot_table(index="avail_date", columns="code", values=field,
                           aggfunc="last", sort=True)
              .reindex(columns=codes))
    idx = pd.Index(sorted(set(dates) | set(wide.index)), name="date")
    wide = wide.reindex(idx).ffill().reindex(pd.Index(dates, name="date"))
    return wide.astype("float32")


def coverage() -> dict:
    init()
    with store.connect() as c:
        row = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT code), COUNT(DISTINCT period), "
            "MIN(period), MAX(period) FROM fundamentals").fetchone()
    return {"rows": row[0] or 0, "codes": row[1] or 0, "periods": row[2] or 0,
            "first_period": row[3], "last_period": row[4]}
