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


# 一期全市场约 5000 多家。低于这个数多半是还没披露完、或抓取被截断了。
MIN_PERIOD_ROWS = 3000


def _period_rows() -> dict[str, int]:
    init()
    with store.connect() as c:
        return dict(c.execute(
            "SELECT period, COUNT(*) FROM fundamentals GROUP BY period").fetchall())


def _newest(ps) -> str | None:
    """point-in-time 意义上最新的一期：先比披露截止日，撞车（年报与次年一季报
    同在 4-30）时取报告期更新的——与 _pit_wide 的取行规则一致。"""
    ps = list(ps)
    return max(ps, key=lambda p: (avail_date(p), p)) if ps else None


def expected_period(today: str | None = None) -> str | None:
    """按法定披露截止日，today 这天应该已经用上的最新一期报告。"""
    return _newest(periods(end=today))


def staleness(today: str | None = None) -> str | None:
    """库里的基本面落后于应有的报告期时返回一句提示，不落后返回 None。

    2026-09 踩过：中报 08-31 就该用上，库里一直停在一季报，growth_value 按
    半年前的财报选了四周股票，不报错。换成中报后 10 只候选换掉 5 只。
    """
    want = expected_period(today)
    if not want:
        return None
    have = [p for p, n in _period_rows().items() if n >= MIN_PERIOD_ROWS]
    if want in have:
        return None
    return (f"基本面最新只到 {_newest(have) or '（无）'}，按法定披露截止日 "
            f"{avail_date(want)} 起应已有 {want}。基本面打分用的是过期财报——"
            f"先跑 sync --fundamentals。")


def sync_recent(progress=None, sleep: float = 0.8) -> dict:
    """只补缺的报告期（已过截止日、但库里不足 MIN_PERIOD_ROWS 行的）。

    全量 sync() 要把 2017 年以来 30 多期挨个抓一遍，每期几十页、一两分钟，
    日常用不着；平时只有季报截止日（4/30、8/31、10/31）之后会冒出一期新的。
    """
    rows = _period_rows()
    todo = [p for p in periods() if rows.get(p, 0) < MIN_PERIOD_ROWS]
    return _sync_periods(todo, progress, sleep)


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
    return _sync_periods(periods(start_year), progress, sleep)


def _sync_periods(ps: list[str], progress=None, sleep: float = 0.8) -> dict:
    init()
    stats = {"periods": len(ps), "ok": 0, "failed": 0, "rows": 0,
             "done": [], "failed_periods": []}
    cols = ["code", "period", "avail_date", "eps", "bps", "roe", "gross_margin",
            "ocfps", "revenue", "revenue_yoy", "profit", "profit_yoy", "industry"]
    for i, p in enumerate(ps, 1):
        try:
            df = fetch_period(p)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            stats["failed"] += 1
            stats["failed_periods"].append(p)
        else:
            df = df.reindex(columns=cols)
            with store.connect() as c:
                c.executemany(
                    f"INSERT OR REPLACE INTO fundamentals ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", store._rows(df))
            stats["ok"] += 1
            stats["rows"] += len(df)
            stats["done"].append(p)
            clear_cache()
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


# ------------------------------------------------------------------ 缓存
# as_panel 是**逐股**调用的：策略的 prepare() 只拿得到单只股票的 df，
# 于是全市场扫一遍就是「股票数 × 字段数」次调用。原先每次都重查一遍库、
# 重做一次 pivot、再对 600 个交易日做一次二维 ffill——实测模拟盘推进一个交易日、
# 150 只股票、2 个字段 = 300 次调用，光 as_panel 就占掉单日耗时的一半以上。
#
# 基本面是季频的：整张宽表只有几十行（每个法定披露截止日一行）。
# 所以把「读库 + pivot + 沿披露日 ffill」全部收进按字段缓存的一步，
# 逐股调用时只剩「取一列 + 按日期二分查找」，与日期序列长度线性相关。
_PIT_LONG: pd.DataFrame | None = None
_PIT_WIDE: dict[str, pd.DataFrame] = {}


def clear_cache() -> None:
    """基本面表被写过之后必须调用，否则同进程内读到的还是旧数据。"""
    global _PIT_LONG
    _PIT_LONG = None
    _PIT_WIDE.clear()
    _BP_IND.clear()


# ------------------------------------------------------------------ 行业内 BP 分位
# BP 跨行业不可比：银行市净率常年零点几，科技股动辄 5 倍以上。全市场统一排百分位，
# 金融地产在「价值」这一半天然高分——2026-09 实测 growth_value 在 800 只池里，
# 金融地产只占 6-11%，买入却占 31-50%，且这部分交易每笔收益两段区间都低于其他行业。
#
# 这里在**全市场同行业**里给每只股票的 BP 排百分位，作为一个普通字段交给引擎。
# 不改引擎/模拟盘/每日信号那三处横截面排名：某天某只股票的这个值只取决于当天
# 同行业的全部股票，任何路径取到的都一样，三处天然一致。
MIN_INDUSTRY = 5              # 行业内当日有效股票少于此数，改用全市场百分位
_BP_IND: dict = {}


def industry_pct(values: pd.DataFrame, industry: pd.Series,
                 min_group: int = MIN_INDUSTRY) -> pd.DataFrame:
    """逐日在行业内排百分位。纯函数，自检直接喂构造数据。

    values：日期 × 股票；industry：股票 -> 行业（缺失归入「未分类」）。
    某行业当日有效值不足 min_group 只时，这些格子改用全市场百分位——
    三五只股票的组内排名基本是噪声，排第一不说明任何事。
    """
    ind = industry.reindex(values.columns).fillna("未分类")
    market = values.rank(axis=1, pct=True)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    for _name, cols in ind.groupby(ind).groups.items():
        sub = values[cols]
        r = sub.rank(axis=1, pct=True)
        small = sub.notna().sum(axis=1) < min_group
        if small.any():
            r.loc[small] = market.loc[small, cols]
        out[cols] = r
    return out


def _latest_industry() -> pd.Series:
    """每只股票最新一期报告里的行业。用最新标签而不是逐期标签：行业变更极少，
    这点前视可以忽略；逐期标签反而会让同一只股票在换期时跳组。"""
    fd = _pit_long()
    if fd.empty or "industry" not in fd.columns:
        return pd.Series(dtype=object)
    f = fd.dropna(subset=["industry"]).sort_values("period")
    return f.groupby("code")["industry"].last()


def _build_bp_industry(lo: str, hi: str) -> pd.DataFrame:
    with store.connect() as c:
        px = pd.read_sql("SELECT code, date, close FROM daily WHERE date BETWEEN ? AND ? "
                         "AND code NOT LIKE 'IDX%'", c, params=(lo, hi))
    if px.empty:
        return pd.DataFrame()
    close = px.pivot(index="date", columns="code", values="close")
    close = close.where(close > 0)          # 前复权负价没有意义，不参与排名
    bps = as_panel("bps", close.index.tolist(), list(close.columns)).astype("float64")
    bp = (bps / close).where(bps > 0)       # 净资产为负的不参与：同 entry 的口径
    return industry_pct(bp, _latest_industry()).astype("float32")


def bp_industry_pct(dates: list[str], code: str) -> np.ndarray:
    """该股每个交易日的「行业内 BP 百分位」（全市场同行业里的名次，0~1）。

    整张面板按请求的日期跨度建一次、进程内缓存；后来的请求落在已建跨度内就直接取列。
    引擎逐股调用时各股的日期跨度基本相同，所以全市场只建一次。
    """
    if not dates:
        return np.array([], dtype="float32")
    lo, hi = dates[0], dates[-1]
    p = _BP_IND.get("panel")
    if p is None or lo < _BP_IND["lo"] or hi > _BP_IND["hi"]:
        lo = min(lo, _BP_IND.get("lo", lo))
        hi = max(hi, _BP_IND.get("hi", hi))
        p = _build_bp_industry(lo, hi)
        _BP_IND.update(panel=p, lo=lo, hi=hi)
    if p.empty or code not in p.columns:
        return np.full(len(dates), np.nan, dtype="float32")
    return p[code].reindex(dates).to_numpy(dtype="float32")


def single_quarter_yoy(fd: pd.DataFrame) -> pd.Series:
    """单季营收同比（%），与 fd 同索引。纯函数，自检直接喂构造数据。

    表里的 revenue 是年初至今累计值，revenue_yoy 也是累计同比：中报的同比里
    混着一季度，三季报的混着前两个季度，最新那个季度的变化被稀释掉了。
    单季 = 本期累计 - 上一期累计（一季度就是累计本身），再与上年同一季度比。

    没有前视：用到的上一期、上年同期都不晚于本期，派生值随本期一起在本期的
    法定截止日生效。缺上一期或上年同期（新股、缺报）时为 NaN；上年同季 <= 0 时
    同比无意义，也记 NaN。
    """
    f = fd[["code", "period", "revenue"]].copy()
    y = pd.to_numeric(f["period"].str[:4])
    m = pd.to_numeric(f["period"].str[4:6])
    cum = pd.Series(f["revenue"].to_numpy(dtype="float64"),
                    index=pd.MultiIndex.from_arrays([f["code"], y, m]))
    cum = cum[~cum.index.duplicated(keep="last")]

    prev_cum = cum.reindex(pd.MultiIndex.from_arrays([f["code"], y, m - 3])).to_numpy()
    sq = np.where(m.to_numpy() == 3, f["revenue"].to_numpy(dtype="float64"),
                  f["revenue"].to_numpy(dtype="float64") - prev_cum)
    sq_idx = pd.Series(sq, index=pd.MultiIndex.from_arrays([f["code"], y, m]))
    sq_idx = sq_idx[~sq_idx.index.duplicated(keep="last")]
    base = sq_idx.reindex(pd.MultiIndex.from_arrays([f["code"], y - 1, m])).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        yoy = np.where(base > 0, (sq / base - 1.0) * 100.0, np.nan)
    return pd.Series(yoy, index=fd.index)


def _pit_long() -> pd.DataFrame:
    global _PIT_LONG
    if _PIT_LONG is None:
        _PIT_LONG = load_pit()
    return _PIT_LONG


def _pit_wide(field: str) -> pd.DataFrame:
    """某字段的 (法定披露截止日 × 全部股票) 宽表，已沿披露日 ffill。整进程只算一次。

    ffill 必须在这里做，不能留到按交易日 reindex 的时候：某一期缺某只股票时，
    宽表那一格是 NaN，而 reindex(method="ffill") 是按**行位置**取值的，
    取到那一行就得到 NaN，不会再往上找。先把宽表本身补齐才是对的。
    """
    w = _PIT_WIDE.get(field)
    if w is None:
        fd = _pit_long()
        if field == "ytd_months" and not fd.empty:
            # 派生字段：该期报告覆盖的月数（营收、利润都是年初至今累计值）。
            # 必须跟着行走、不能按交易日推算：某公司当期缺报时 ffill 会拿到更早一期，
            # 按日期推出来的月数就对不上（例：缺一季报时拿到的是 12 个月的年报）。
            fd = fd.assign(ytd_months=pd.to_numeric(fd["period"].str[4:6]))
        elif field == "revenue_q_yoy" and not fd.empty:
            fd = fd.assign(revenue_q_yoy=single_quarter_yoy(fd))
        if fd.empty or field not in fd.columns:
            w = pd.DataFrame()
        else:
            f = fd[["code", "period", "avail_date", field]].dropna(subset=[field])
            # 年报(1231)与次年一季报(0331)的法定截止日**都是 4-30**，会撞在同一天。
            # 撞车时必须取更新的那期（一季报），所以先按 period 升序，再用 last。
            f = f.sort_values(["code", "avail_date", "period"])
            w = f.pivot_table(index="avail_date", columns="code", values=field,
                              aggfunc="last", sort=True).ffill()
        _PIT_WIDE[field] = w
    return w


def as_panel(field: str, dates: list[str],
             codes: list[str]) -> pd.DataFrame:
    """把某个基本面字段展开成 point-in-time 宽表（日期 × 股票）。

    每个交易日取**该日之前已过披露截止日**的最新一期数据。
    这样 2025-01-15 这天看到的仍是 2024 年三季报，直到 2025-04-30 才切到年报——
    与真实世界的信息可得性一致。
    """
    wide = _pit_wide(field)
    idx = pd.Index(dates, name="date")
    if wide.empty:
        return pd.DataFrame(index=idx, columns=codes, dtype="float32")
    # 先取列再对齐日期：逐股调用时这里只有一列，代价与股票数无关。
    # method="ffill" 走的是二分查找，不会像二维 ffill 那样铺满整个日期 × 股票矩阵。
    sub = wide.reindex(columns=codes)
    return sub.reindex(idx, method="ffill").astype("float32")


def coverage() -> dict:
    init()
    with store.connect() as c:
        row = c.execute(
            "SELECT COUNT(*), COUNT(DISTINCT code), COUNT(DISTINCT period), "
            "MIN(period), MAX(period) FROM fundamentals").fetchone()
    return {"rows": row[0] or 0, "codes": row[1] or 0, "periods": row[2] or 0,
            "first_period": row[3], "last_period": row[4]}
