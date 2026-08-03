"""行情抓取：封装 akshare，做列名归一化、并发、重试和增量同步。

akshare 是爬虫聚合库，接口列名会随上游改动，所以这里全部走"模糊列名映射"，
少一两列不至于让整条同步链路挂掉。
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from ..config import HISTORY_START, classify_board
from . import store

# 并发度。曾经用 24 只股票的小样本测出「12 线程最快且成功率最高」，据此设成 12——
# 那个结论是错的。一次 4429 只的全市场同步暴露了真相：失败率随时间单调恶化，
# 前 3150 只失败 20%，最后 200 只失败 92%。这是**累积触发的限流**，不是随机抖动，
# 短样本根本测不出来。降并发 + 请求间隔 + 失败后长冷却才是对症的做法。
MAX_WORKERS = 5
REQUEST_GAP = 0.15       # 每个请求前的最小间隔（秒），给上游降速
RETRY = 4
RETRY_SLEEP = 2.0        # 退避基数，实际为 RETRY_SLEEP * 2^n + 抖动

# 增量同步时往回重抓的天数，用于覆盖盘中写入的残缺 K 线
OVERLAP_DAYS = 7


def _ak():
    import akshare as ak
    return ak


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """按候选名依次找列，找不到返回 None。"""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


# ------------------------------------------------------------------ 股票列表
def _num(s: pd.Series | None) -> pd.Series | None:
    """交易所返回的股本带千分位逗号，先去掉再转数字。"""
    if s is None:
        return None
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def fetch_instruments() -> pd.DataFrame:
    """拉取全 A 股票列表。

    主源用沪深交易所官方接口：单次请求、稳定，且带上市日期和行业，
    比东财全市场快照（要翻 50+ 页、容易被限流）可靠得多。
    """
    ak = _ak()
    frames: list[pd.DataFrame] = []

    # 上交所
    try:
        sh = ak.stock_info_sh_name_code()
        f = pd.DataFrame({
            "code": _pick(sh, "证券代码", "SECURITY_CODE_A").astype(str).str.zfill(6),
            "name": _pick(sh, "证券简称", "SECURITY_ABBR_A").astype(str).str.strip(),
            "listed_date": _pick(sh, "上市日期", "LISTING_DATE").astype(str),
        })
        f["industry"] = pd.NA
        f["float_share"] = pd.NA
        frames.append(f)
    except Exception as e:
        print(f"[warn] 上交所列表获取失败：{e}")

    # 深交所（额外带行业和流通股本）
    try:
        sz = ak.stock_info_sz_name_code()
        f = pd.DataFrame({
            "code": _pick(sz, "A股代码").astype(str).str.zfill(6),
            "name": _pick(sz, "A股简称").astype(str).str.strip(),
            "listed_date": _pick(sz, "A股上市日期").astype(str),
            "industry": _pick(sz, "所属行业"),
            "float_share": _num(_pick(sz, "A股流通股本")),
        })
        frames.append(f)
    except Exception as e:
        print(f"[warn] 深交所列表获取失败：{e}")

    if not frames:
        raise RuntimeError("沪深两市股票列表都获取失败，请检查网络/代理设置")

    out = pd.concat(frames, ignore_index=True)
    # 深市简称里有全角空格（如 "万  科Ａ"），统一压掉
    out["name"] = out["name"].str.replace(r"\s+", "", regex=True)
    out["listed_date"] = pd.to_datetime(
        out["listed_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    out["board"] = out["code"].map(classify_board)
    out["is_st"] = out["name"].str.upper().str.contains("ST").astype(int)
    out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out = out[out["board"] != "未知"]
    return out.drop_duplicates(subset=["code"]).dropna(subset=["code", "name"])


def sync_instruments() -> int:
    df = fetch_instruments()
    n = store.upsert_instruments(df)
    store.set_meta("instruments_synced_at", datetime.now().isoformat(timespec="seconds"))
    return n


# ------------------------------------------------------------------ 日线
_HIST_COLS = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
}


def fetch_daily(code: str, start: str = HISTORY_START, end: str | None = None,
                adjust: str = "qfq") -> pd.DataFrame:
    """单只股票日线（默认前复权）。失败重试，最终失败返回空表而非抛出。"""
    ak = _ak()
    end = end or datetime.now().strftime("%Y%m%d")
    for attempt in range(RETRY):
        try:
            if REQUEST_GAP:
                time.sleep(REQUEST_GAP * (0.5 + random.random()))
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
            if raw is None or raw.empty:
                return pd.DataFrame()
            df = raw.rename(columns=_HIST_COLS)
            keep = ["date", "open", "high", "low", "close",
                    "volume", "amount", "pct_chg", "turnover"]
            df = df.reindex(columns=keep)
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for c in keep[1:]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df.dropna(subset=["close"])
        except Exception:
            if attempt == RETRY - 1:
                return pd.DataFrame()
            # 指数退避 + 抖动：被限流时一起重试会再次把上游打满
            time.sleep(RETRY_SLEEP * (2 ** attempt) * (0.5 + random.random()))
    return pd.DataFrame()


def market_closed_today() -> bool:
    """今天的收盘时段是否已结束（A 股 15:00 收盘，留 5 分钟余量）。

    用本机本地时间判断，假定机器时区为北京时间。
    """
    now = datetime.now()
    return now.hour * 60 + now.minute >= 15 * 60 + 5


def sync_daily(codes: list[str] | None = None, full: bool = False,
               progress=None, overlap_days: int = OVERLAP_DAYS,
               only_missing: bool = False) -> dict:
    """增量同步日线。

    增量更新不是从"本地最后日期 + 1 天"开始，而是**往回退 overlap_days 天重新抓**，
    因为盘中同步会把当天没走完的 K 线写进库里（成交量只有半天的量），
    若下次只从次日补起，这根残缺 K 线就永远留在库里污染所有回测。
    upsert 是 INSERT OR REPLACE，重叠部分会被正确的完整数据覆盖。

    前复权价格还会随分红送转整体变化，所以仍建议定期跑一次 full=True 重建全部历史。
    """
    store.init_db()
    inst = store.load_instruments()
    if inst.empty:
        sync_instruments()
        inst = store.load_instruments()

    if codes is None:
        codes = inst["code"].tolist()

    last = {} if full else store.last_dates()
    today = datetime.now().strftime("%Y-%m-%d")

    # 只有当"本地已有今天的数据"且"上次同步发生在今天收盘之后"时，才认为数据已是最终版，
    # 可以整只跳过。盘中同步写下的当日 K 线不满足后一条，下次会被重新抓取覆盖。
    synced_at = store.get_meta("daily_synced_at") or ""
    fresh = bool(synced_at >= f"{today}T15:05") and market_closed_today()

    pending: list[tuple[str, str]] = []
    for code in codes:
        if full or code not in last:
            pending.append((code, HISTORY_START))
        else:
            # only_missing：只补从没取到过的股票，已有数据的一律跳过。
            # 用于上游限流后专门补缺——此时重拉几千只已有的股票既慢又会再次触发限流。
            if only_missing or (last[code] >= today and fresh):
                continue
            # 往回退若干天重抓，覆盖可能残缺的当日/近日 K 线
            back = (pd.to_datetime(last[code]) - timedelta(days=overlap_days))
            pending.append((code, back.strftime("%Y%m%d")))

    stats = {"requested": len(codes), "pending": len(pending),
             "ok": 0, "failed": 0, "rows": 0, "passes": 0}
    if not pending:
        return stats

    total = len(pending)
    done = 0

    def _sweep(batch: list[tuple[str, str]], workers: int) -> list[tuple[str, str]]:
        """跑一遍，返回失败的（供下一轮重试）。"""
        nonlocal done
        failed: list[tuple[str, str]] = []
        starts = dict(batch)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_daily, c, s): c for c, s in batch}
            for fut in as_completed(futures):
                code = futures[fut]
                done += 1
                try:
                    df = fut.result()
                except Exception:
                    df = pd.DataFrame()
                if df.empty:
                    failed.append((code, starts[code]))
                else:
                    stats["rows"] += store.upsert_daily(df)
                    stats["ok"] += 1
                if progress and done % 50 == 0:
                    # failed 只能按"已尝试 - 已成功"算，不能用 total - ok，
                    # 否则会把还没轮到的股票也算成失败
                    stats["failed"] = done - stats["ok"]
                    progress(done, total, stats)
        return failed

    # 多轮：每轮之后冷却，并进一步降低并发。上游的限流是累积触发的，
    # 一轮扫完时往往已经处于被限速状态，此时立刻重试注定失败——必须先等它恢复。
    remaining = pending
    for pass_no, (workers, cooldown) in enumerate(
            [(MAX_WORKERS, 0), (3, 60), (2, 120)], start=1):
        if not remaining:
            break
        if cooldown:
            if progress:
                stats["failed"] = len(remaining)
                progress(done, total, {**stats, "phase": f"冷却 {cooldown}s 后重试"})
            time.sleep(cooldown)
            total += len(remaining)          # 重试的这些会被再数一遍
        stats["passes"] = pass_no
        remaining = _sweep(remaining, workers)

    stats["failed"] = len(remaining)
    stats["failed_codes"] = [c for c, _ in remaining]
    store.set_meta("daily_synced_at", datetime.now().isoformat(timespec="seconds"))
    return stats


# ------------------------------------------------------------------ 退市股
def _retry(fn, *a, tries: int = 4, **kw):
    """交易所接口偶发 SSL 断连，退避重试。"""
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(RETRY_SLEEP * (2 ** i) * (0.5 + random.random()))


def fetch_delisted() -> pd.DataFrame:
    """已退市股票列表。

    这是修正幸存者偏差的关键：交易所的在市股票列表里没有退市股，
    只用它建股票池，等于每次都只在"活下来的公司"里选——
    历史回测会系统性偏乐观，小市值类因子受影响尤其大。
    """
    ak = _ak()
    frames = []
    try:
        sh = _retry(ak.stock_info_sh_delist)
        frames.append(pd.DataFrame({
            "code": sh["公司代码"].astype(str).str.zfill(6),
            "name": sh["公司简称"].astype(str).str.strip(),
            "listed_date": sh["上市日期"].astype(str),
            "delisted_date": sh["暂停上市日期"].astype(str),
        }))
    except Exception as e:
        print(f"[warn] 上交所退市列表获取失败：{e}")
    try:
        sz = _retry(ak.stock_info_sz_delist, symbol="终止上市公司")
        frames.append(pd.DataFrame({
            "code": sz["证券代码"].astype(str).str.zfill(6),
            "name": sz["证券简称"].astype(str).str.strip(),
            "listed_date": sz["上市日期"].astype(str),
            "delisted_date": sz["终止上市日期"].astype(str),
        }))
    except Exception as e:
        print(f"[warn] 深交所退市列表获取失败：{e}")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["name"] = out["name"].str.replace(r"\s+", "", regex=True)
    for c in ("listed_date", "delisted_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out["board"] = out["code"].map(classify_board)
    out["is_st"] = 1                      # 退市股基本都经历过 ST
    out["industry"] = pd.NA
    out["float_share"] = pd.NA
    out["status"] = "delisted"
    out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = out[out["board"] != "未知"]
    return out.dropna(subset=["code", "delisted_date"]).drop_duplicates(subset=["code"])


def sync_delisted(since: str = HISTORY_START, progress=None) -> dict:
    """同步退市股的基础信息与历史日线。

    只取在本地数据窗口内还活着的（退市日 >= since）——更早退市的公司
    在我们的回测区间里本来就不存在，取了也没用。
    """
    store.init_db()
    lst = fetch_delisted()
    if lst.empty:
        return {"listed": 0, "synced": 0, "rows": 0}

    since_iso = pd.to_datetime(since).strftime("%Y-%m-%d")
    lst = lst[lst["delisted_date"] >= since_iso]
    store.upsert_instruments(lst)

    codes = lst["code"].tolist()
    have = store.last_dates()
    todo = [c for c in codes if c not in have]
    stats = {"listed": len(lst), "pending": len(todo), "synced": 0,
             "failed": 0, "rows": 0}
    if not todo:
        return stats

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(fetch_daily, c, since): c for c in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                df = fut.result()
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                stats["failed"] += 1
            else:
                stats["rows"] += store.upsert_daily(df)
                stats["synced"] += 1
            if progress and done % 20 == 0:
                progress(done, len(todo), stats)
    return stats


# ------------------------------------------------------------------ 指数（基准）
# key 为带交易所前缀的新浪代码，value 为 (显示名, 本地存储代码)
BENCHMARKS = {
    "sh000300": ("沪深300", "IDX000300"),
    "sh000905": ("中证500", "IDX000905"),
    "sh000001": ("上证指数", "IDX000001"),
    "sz399006": ("创业板指", "IDX399006"),
}


def sync_index(symbol: str = "sh000300", start: str = HISTORY_START) -> int:
    """同步指数日线，存进同一张 daily 表，回测时当基准用。

    用新浪源（stock_zh_index_daily）而非东财的 index_zh_a_hist：后者内部要先翻十几页
    拉全指数列表才能查单个指数，很容易被限流；新浪是单次请求且能回溯到 2002 年。

    指数代码会与个股代码撞车（000001 既是上证指数也是平安银行），故统一加 IDX 前缀存储。
    """
    ak = _ak()
    local_code = BENCHMARKS.get(symbol, (symbol, f"IDX{symbol[-6:]}"))[1]
    try:
        raw = ak.stock_zh_index_daily(symbol=symbol)
    except Exception as e:
        print(f"[warn] 指数 {symbol} 获取失败：{e}")
        return 0
    if raw is None or raw.empty:
        return 0

    df = pd.DataFrame({
        "code": local_code,
        "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "volume": pd.to_numeric(raw["volume"], errors="coerce"),
    })
    df["amount"] = pd.NA
    df["pct_chg"] = df["close"].pct_change() * 100
    df["turnover"] = pd.NA
    start_iso = pd.to_datetime(start).strftime("%Y-%m-%d")
    df = df[df["date"] >= start_iso].dropna(subset=["close"])
    return store.upsert_daily(df)


# ------------------------------------------------------------------ 实时快照
def _spot_one(code: str) -> dict | None:
    """单只股票最新价。

    用日线接口取最近几天的最后一根 K 线，而不是专门的实时报价接口：
    stock_bid_ask_em 等接口经常失效，而日线接口是整套系统里最可靠的一个。
    盘中该接口返回的当日 K 线，收盘价字段即为当前最新价。

    关键：这里必须用**不复权**价（adjust=""）。持仓成本是你实际成交的价格，
    拿前复权价去比会算出错误的盈亏——除权后前复权价会整体下移。
    """
    start = (datetime.now() - timedelta(days=12)).strftime("%Y%m%d")
    df = fetch_daily(code, start=start, adjust="")
    if df is None or df.empty:
        return None
    row = df.sort_values("date").iloc[-1]
    return {
        "code": code,
        "price": float(row["close"]),
        "pct_chg": float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else None,
        "date": row["date"],
    }


def fetch_spot(codes: list[str]) -> pd.DataFrame:
    """取最新价，用于界面显示持仓浮盈。

    逐只并发请求而非拉全市场快照——东财全市场接口要翻 50+ 页很容易被限流，
    而我们通常只关心手上十几只票。取不到的由调用方回退到本地最近收盘价。
    """
    codes = list(codes)
    cols = ["code", "price", "pct_chg", "date"]
    if not codes:
        return pd.DataFrame(columns=cols)
    rows = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(codes))) as pool:
        for res in pool.map(_spot_one, codes):
            if res:
                rows.append(res)
    return pd.DataFrame(rows, columns=cols)
