"""行情抓取：封装 akshare，做列名归一化、并发、重试和增量同步。

akshare 是爬虫聚合库，接口列名会随上游改动，所以这里全部走"模糊列名映射"，
少一两列不至于让整条同步链路挂掉。
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

from ..config import HISTORY_START, classify_board
from . import store

# ------------------------------------------------------------------ 熔断器
# 某个数据源连续失败 N 次后，临时跳过该源，直接切到下一个，
# 避免在已经挂掉的源上死等。
#
# 必须有「半开」恢复：熔断后该源被跳过，跳过就不会有成功，也就永远等不到重置——
# 没有超时恢复的话，一次瞬时失败会把这个源**永久废掉**（实测四源各失败一次后
# fetch_daily 对所有股票直接返回空表）。所以这里记录熔断时刻，
# 冷却期一过就放行一次试探，成功即完全恢复。
_circuit_state: dict[str, int] = {"tencent": 0, "baostock": 0,
                                  "tushare": 0, "akshare": 0}
_circuit_until: dict[str, float] = {}
_circuit_lock = threading.Lock()
CIRCUIT_THRESHOLD = 3      # 连续失败达到此数才熔断（1 太敏感，网络抖动就会误伤）
CIRCUIT_COOLDOWN = 60.0    # 冷却秒数，过后放行一次试探


def _circuit_open(src: str) -> bool:
    """该源当前是否应跳过。冷却期已过则放行试探。"""
    with _circuit_lock:
        if _circuit_state.get(src, 0) < CIRCUIT_THRESHOLD:
            return False
        if time.time() >= _circuit_until.get(src, 0.0):
            # 半开：允许一次试探。失败会重新计时，成功则由 _circuit_reset 清零
            _circuit_until[src] = time.time() + CIRCUIT_COOLDOWN
            return False
        return True


def _circuit_fail(src: str) -> None:
    """记录一次源失败，达到阈值则开始计时冷却。"""
    with _circuit_lock:
        n = _circuit_state.get(src, 0) + 1
        _circuit_state[src] = n
        if n >= CIRCUIT_THRESHOLD:
            _circuit_until[src] = time.time() + CIRCUIT_COOLDOWN


def _circuit_reset(src: str) -> None:
    """源成功一次，完全恢复。"""
    with _circuit_lock:
        _circuit_state[src] = 0
        _circuit_until.pop(src, None)

# 并发度的历史教训：曾用 24 只股票的小样本测出「12 线程最快」，据此设成 12——
# 那是错的。一次 4429 只的单源全市场同步里，失败率随时间单调恶化（前 3150 只失败
# 20%，最后 200 只失败 92%），是**累积触发的限流**，短样本测不出来。
#
# 多源 fallback 确实能分摊压力，理论上可以提高并发；但唯一一次实测有据的
# 全市场同步是「并发 5 + 间隔 0.15s」跑出的 4589 只成功、失败 0。
# 在没有同等规模的实测支撑之前，保守值优先——一次干净的全量同步
# 比快一倍但要重跑三遍划算。
MAX_WORKERS = 5
REQUEST_GAP = 0.15       # 每个请求前的最小间隔（秒）
RETRY = 2                # 单源重试次数——失败快速切下一个源，不在此源上死等
SINGLE_SOURCE_RETRY = 2
RETRY_SLEEP = 1.0        # 退避基数，实际为 RETRY_SLEEP * 2^n + 抖动

# 增量同步时往回重抓的天数，用于覆盖盘中写入的残缺 K 线
OVERLAP_DAYS = 7

# BaoStock 默认关闭：它的前复权序列与腾讯/akshare **不兼容**。
#
# 实测（2024-01 ~ 2026-07，500 个重叠交易日）：
#   腾讯 vs akshare —— 价格比值恒为 1.0000，日收益率最大差 3e-06（纯浮点噪声）
#   baostock vs 腾讯 —— 比值极差 0.025~0.048，最大价格偏差 1.4%~5.1%，
#                       日收益率差 >0.1% 的天数占 10%~23%
#
# 如果只是复权锚点不同，比值会是恒定常数、收益率完全一致；比值有极差
# 说明复权算法本身不同。多源 fallback 的前提是各源可互换，baostock 不满足：
# 同一只股票半段来自腾讯、半段来自 baostock，会在切换点产生**假的跳空**，
# 而指标会把它当成真实的暴涨暴跌——属于"不报错但让回测失真"的那类问题。
#
# 想启用请自行确认复权口径一致，或只用它同步本地完全没有数据的新股票。
USE_BAOSTOCK = False


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


def _fetch_akshare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """akshare 东财源：单只股票日线。失败返回空表。"""
    ak = _ak()
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
            time.sleep(RETRY_SLEEP * (2 ** attempt) * (0.5 + random.random()))
    return pd.DataFrame()


TENCENT_MAX_BARS = 500          # 腾讯单次请求的硬上限


def _fetch_tencent(code: str, start: str, end: str,
                   adjust: str = "qfq") -> pd.DataFrame:
    """腾讯数据源：单只股票日线，自动分段绕过单次 500 根的上限。

    走 web.ifzq.gtimg.cn 接口，绕开东财 CDN 的 IP 封禁，免费无需 token。

    单次请求最多返回 500 根 K 线。曾经因为没处理这个上限，
    请求 2018-2026 只拿回最近 500 根（2024-07 起），
    更早的数据保持不变——在本地库里表现为「2024-07 突然断层」。
    """
    spans, cur = [], pd.to_datetime(end)
    lo = pd.to_datetime(start)
    # 500 个交易日约合 730 个自然日，留余量按 680 天一段往回切
    while cur >= lo:
        seg_start = max(lo, cur - pd.Timedelta(days=680))
        spans.append((seg_start.strftime("%Y-%m-%d"), cur.strftime("%Y-%m-%d")))
        if seg_start <= lo:
            break
        cur = seg_start - pd.Timedelta(days=1)

    parts = [_fetch_tencent_span(code, a, b, adjust) for a, b in spans]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    out = (pd.concat(parts, ignore_index=True)
             .drop_duplicates(subset=["date"])
             .sort_values("date").reset_index(drop=True))
    # 分段拼接后 pct_chg 在段边界会断，统一重算一次
    out["pct_chg"] = out["close"].pct_change() * 100
    return out


def _fetch_tencent_span(code: str, start: str, end: str,
                        adjust: str = "qfq") -> pd.DataFrame:
    """腾讯源单段请求（<= 500 根）。"""
    import json
    import urllib.request

    # 腾讯接口需要带交易所前缀的代码：sh600519 / sz000001
    qfq = "qfq" if adjust in ("qfq", "q") else ""
    if code.startswith(("6", "9")):
        tcode = f"sh{code}"
    else:
        tcode = f"sz{code}"

    start_iso = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_iso = pd.to_datetime(end).strftime("%Y-%m-%d")
    url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"param={tcode},day,{start_iso},{end_iso},{TENCENT_MAX_BARS},{qfq}")

    for attempt in range(RETRY):
        try:
            if REQUEST_GAP:
                time.sleep(REQUEST_GAP * (0.5 + random.random()))
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36",
                "Referer": "https://web.ifzq.gtimg.cn/",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            stock_data = data.get("data", {})
            if not stock_data:
                return pd.DataFrame()
            stock_key = next(iter(stock_data), None)
            if not stock_key:
                return pd.DataFrame()
            # 前复权时优先取 qfqday，否则取 day
            klines = (stock_data[stock_key].get("qfqday")
                      if adjust in ("qfq", "q")
                      else stock_data[stock_key].get("day"))
            if not klines:
                return pd.DataFrame()
            rows = []
            for k in klines:
                if len(k) >= 6:
                    rows.append({
                        "date": k[0],
                        "open": float(k[1]),
                        "close": float(k[2]),
                        "high": float(k[3]),
                        "low": float(k[4]),
                        "volume": float(k[5]),
                    })
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)

            # 腾讯的 K 线只有 6 个字段（日期/开/收/高/低/量），没有成交额、
            # 涨跌幅、换手率。直接置空会造成静默灾难：universe.build 用
            # 20 日均成交额筛流动性，amount 全 NULL 会让股票池只剩十几只，
            # 2024-07 之后的回测全部返回空——而且不报任何错。所以必须补算。
            #
            # pct_chg 由收盘价精确推出（同一复权序列，无误差）。
            # amount 只能估算：成交量(手) × 100 × 均价，用收盘价代理均价。
            # 实测 amount/(volume×close) ≈ 100，说明这个代理够准；
            # 而它只用于「是否超过 5000 万」这类流动性门槛，几个百分点的
            # 误差无关紧要。turnover 需要流通股本，这里给不出，保持为空。
            df["pct_chg"] = df["close"].pct_change() * 100
            df["amount"] = df["volume"] * 100.0 * df["close"]
            df["turnover"] = pd.NA
            return df.dropna(subset=["close"])
        except Exception:
            if attempt == RETRY - 1:
                return pd.DataFrame()
            time.sleep(RETRY_SLEEP * (2 ** attempt) * (0.5 + random.random()))
    return pd.DataFrame()


def _fetch_baostock(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """BaoStock 数据源：单只股票日线。

    移植自 Vibe-Trading 的 baostock_loader：走 TCP 协议，绕开 HTTP 数据源
    （东财/腾讯）的 CDN 封禁。免费、无需 token。返回与 akshare 相同的列结构。
    """
    import baostock as bs

    # baostock 需要带交易所前缀：sh.600519 / sz.000001
    # adjustflag: 1 = 后复权, 2 = 前复权, 3 = 不复权
    adj_flag = "1" if adjust in ("hfq", "h") else ("3" if adjust in ("", "none") else "2")
    if code.startswith(("6", "9")):
        bs_code = f"sh.{code}"
    else:
        bs_code = f"sz.{code}"

    start_iso = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_iso = pd.to_datetime(end).strftime("%Y-%m-%d")

    for attempt in range(RETRY):
        try:
            if REQUEST_GAP:
                time.sleep(REQUEST_GAP * (0.5 + random.random()))
            lg = bs.login()
            if lg.error_code != "0":
                return pd.DataFrame()
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_iso, end_date=end_iso,
                    frequency="d", adjustflag=adj_flag,
                )
                if rs.error_code != "0":
                    return pd.DataFrame()
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    return pd.DataFrame()
                df = pd.DataFrame(rows, columns=[
                    "date", "open", "high", "low", "close", "volume", "amount"])
                df["code"] = code
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                for c in ("open", "high", "low", "close", "volume", "amount"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["pct_chg"] = pd.NA
                df["turnover"] = pd.NA
                return df.dropna(subset=["close"])
            finally:
                bs.logout()
        except Exception:
            if attempt == RETRY - 1:
                return pd.DataFrame()
            time.sleep(RETRY_SLEEP * (2 ** attempt) * (0.5 + random.random()))
    return pd.DataFrame()


def _to_tushare_code(code: str) -> str:
    """6 位代码 → tushare 格式（600519.SH / 000001.SZ）。"""
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _fetch_tushare(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    """Tushare Pro 数据源：单只股票日线（前复权）。

    复用 tushare_source.fetch_daily_qfq，合并 daily + adj_factor 合成前复权。
    需要 TUSHARE_TOKEN（环境变量或 .tushare_token 文件）。
    返回与 akshare 相同的列结构。
    """
    from .tushare_source import fetch_daily_qfq, available
    if not available():
        return pd.DataFrame()
    ts_code = _to_tushare_code(code)
    start_iso = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_iso = pd.to_datetime(end).strftime("%Y-%m-%d")
    for attempt in range(RETRY):
        try:
            if REQUEST_GAP:
                time.sleep(REQUEST_GAP * (0.5 + random.random()))
            df = fetch_daily_qfq(ts_code, start=start_iso)
            if df is None or df.empty:
                return pd.DataFrame()
            # 截取到 end 日期
            df = df[df["date"] <= end_iso]
            if df.empty:
                return pd.DataFrame()
            df["code"] = code
            for c in ("open", "high", "low", "close", "volume", "amount"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df["pct_chg"] = pd.NA
            df["turnover"] = pd.NA
            return df.dropna(subset=["close"])
        except Exception:
            if attempt == RETRY - 1:
                return pd.DataFrame()
            time.sleep(RETRY_SLEEP * (2 ** attempt) * (0.5 + random.random()))
    return pd.DataFrame()


def fetch_daily(code: str, start: str = HISTORY_START, end: str | None = None,
                adjust: str = "qfq", on_event=None) -> pd.DataFrame:
    """单只股票日线（默认前复权）。多源自动 fallback。

    顺序：腾讯 → BaoStock → Tushare → akshare（东财）。快稳的源前置，
    akshare 作为最后兜底（它能提供最全字段，缺点是最慢、最易限流）。
    前一个源失败快速切下一个，全部失败才返回空表并记录 sync_errors。

    熔断机制：某个源连续失败 >= CIRCUIT_THRESHOLD 次后临时跳过该源，冷却后自动恢复，
    直接切到下一个，避免在已经挂掉的源上死等。

    on_event: 可选回调 on_event(source, rows)，同步过程实时上报每只股票来自哪个源、
    取了多少行，供看板显示实时日志。
    """
    end = end or datetime.now().strftime("%Y%m%d")
    start_compact = pd.to_datetime(start).strftime("%Y%m%d")
    end_compact = pd.to_datetime(end).strftime("%Y%m%d")

    # 1) 腾讯（最快，绕开东财 CDN）
    if not _circuit_open("tencent"):
        df = _fetch_tencent(code, start, end, adjust)
        if not df.empty:
            _circuit_reset("tencent")
            if on_event:
                on_event("tencent", len(df))
            return df
        _circuit_fail("tencent")

    # 2) BaoStock —— **默认不启用**，见 USE_BAOSTOCK 的说明
    if USE_BAOSTOCK and not _circuit_open("baostock"):
        df = _fetch_baostock(code, start, end, adjust)
        if not df.empty:
            _circuit_reset("baostock")
            if on_event:
                on_event("baostock", len(df))
            return df
        _circuit_fail("baostock")

    # 3) Tushare Pro（需要 token，退市股专用也能兜底）
    if not _circuit_open("tushare"):
        df = _fetch_tushare(code, start, end, adjust)
        if not df.empty:
            _circuit_reset("tushare")
            if on_event:
                on_event("tushare", len(df))
            return df
        _circuit_fail("tushare")

    # 4) akshare（东财，最全字段但慢）
    if not _circuit_open("akshare"):
        df = _fetch_akshare(code, start_compact, end_compact, adjust)
        if not df.empty:
            _circuit_reset("akshare")
            if on_event:
                on_event("akshare", len(df))
            return df
        _circuit_fail("akshare")

    # 全部失败：记录原因
    if on_event:
        on_event("none", 0)
    store.log_sync_error(code, "tencent/baostock/tushare/akshare 全部失败", attempt=RETRY)
    return pd.DataFrame()


def market_closed_today() -> bool:
    """今天的收盘时段是否已结束（A 股 15:00 收盘，留 5 分钟余量）。

    用本机本地时间判断，假定机器时区为北京时间。
    """
    now = datetime.now()
    return now.hour * 60 + now.minute >= 15 * 60 + 5


def sync_daily(codes: list[str] | None = None, full: bool = False,
               progress=None, overlap_days: int = OVERLAP_DAYS,
               only_missing: bool = False, on_event=None,
               circuit_breaker: int = CIRCUIT_THRESHOLD,
               cancel_check=None) -> dict:
    """增量同步日线。

    增量更新不是从"本地最后日期 + 1 天"开始，而是**往回退 overlap_days 天重新抓**，
    因为盘中同步会把当天没走完的 K 线写进库里（成交量只有半天的量），
    若下次只从次日补起，这根残缺 K 线就永远留在库里污染所有回测。
    upsert 是 INSERT OR REPLACE，重叠部分会被正确的完整数据覆盖。

    前复权价格还会随分红送转整体变化，所以仍建议定期跑一次 full=True 重建全部历史。

    circuit_breaker: 某个源连续失败 N 次后临时跳过（默认 3）。冷却 60 秒后放行试探。
    cancel_check: 可选回调，返回 True 时中断同步。
    """
    global CIRCUIT_THRESHOLD
    CIRCUIT_THRESHOLD = max(1, int(circuit_breaker))
    # 每次同步开始时清空熔断状态，包括冷却计时——否则上一轮遗留的熔断
    # 会让这一轮一上来就跳过某些源
    with _circuit_lock:
        for k in _circuit_state:
            _circuit_state[k] = 0
        _circuit_until.clear()

    store.init_db()
    inst = store.load_instruments()
    if inst.empty:
        sync_instruments()
        inst = store.load_instruments()

    if codes is None:
        codes = inst["code"].tolist()

    # only_missing：只补**本地完全没有数据**的股票，已有数据的一律跳过。
    #
    # 曾一度改成"只补 sync_errors 表里记录过的失败股票"，那是个退化：
    # 失败记录只在同步过程中产生，新库或换机器后该表是空的，
    # 此时 --only-missing 会什么都不做，而这恰恰是最需要它的场景。
    # 以"本地有没有数据"为准则不依赖任何历史状态，任何时候都正确。
    #
    # sync_errors 仍然保留，用于界面展示哪些股票失败过、失败原因是什么。
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
        elif only_missing:
            # 本地已有数据 → 跳过。这正是 --only-missing 的意义：
            # 补缺时重拉几千只已有的股票既慢又会再次触发限流，
            # 实测补 1268 只缺口时能省掉 72% 的请求。
            continue
        else:
            # 增量同步：本地已有今天的数据且上次同步在收盘后，整只跳过。
            if last[code] >= today and fresh:
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
            def _wrap(code, start):
                def _cb(src, rows):
                    if on_event:
                        on_event(code, src, rows)
                return fetch_daily(code, start=start, on_event=_cb)
            futures = {pool.submit(_wrap, c, s): c for c, s in batch}
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
                    store.clear_sync_errors([code])   # 补上了就清掉失败记录
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
        # 检查取消信号
        if cancel_check and cancel_check():
            stats["failed"] = len(remaining)
            stats["failed_codes"] = [c for c, _ in remaining]
            stats["cancelled"] = True
            return stats
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
