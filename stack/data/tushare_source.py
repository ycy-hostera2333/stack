"""Tushare Pro 数据源：专门用于补齐**退市股**历史行情。

为什么需要它：免费零售源（东财/新浪/通达信本地）都不提供退市股的历史 K 线，
实测 367 只退市股在三个源里全部取不到。而 2019 年起跑的回测，
当时约 17% 的可选标的最终退市——它们缺席会让所有历史结论系统性偏乐观，
小市值类因子受影响尤其大。

Token 从环境变量或本地文件读取，**绝不写进代码库**：
    环境变量  TUSHARE_TOKEN
    或文件    <项目根>/.tushare_token   （已在 .gitignore 中）

注册地址 https://tushare.pro ，注册即送 120 积分。
注意 daily 接口需要 120 积分，退市股基础信息需要更高积分，
具体门槛以官网为准——积分不够时本模块会给出明确提示而非静默失败。
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd

from ..config import HISTORY_START, ROOT, classify_board
from . import store

TOKEN_FILE = ROOT / ".tushare_token"
_pro = None


def get_token() -> str | None:
    tok = os.environ.get("TUSHARE_TOKEN", "").strip()
    if tok:
        return tok
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return None


def available() -> bool:
    return get_token() is not None


def _api():
    """惰性初始化，没有 token 时给出可操作的提示。"""
    global _pro
    if _pro is not None:
        return _pro
    tok = get_token()
    if not tok:
        raise RuntimeError(
            "未配置 Tushare token。到 https://tushare.pro 注册后，二选一：\n"
            "  1) 设环境变量  TUSHARE_TOKEN=你的token\n"
            f"  2) 写入文件    {TOKEN_FILE}\n"
            "该文件已在 .gitignore 中，不会被提交。")
    import tushare as ts
    ts.set_token(tok)
    _pro = ts.pro_api()
    return _pro


def _call(name: str, tries: int = 3, **kw) -> pd.DataFrame:
    """调用接口并处理限频。积分不足的报错原样抛出，不吞掉。"""
    pro = _api()
    last = None
    for i in range(tries):
        try:
            return getattr(pro, name)(**kw)
        except Exception as e:
            msg = str(e)
            last = e
            if "积分" in msg or "权限" in msg or "permission" in msg.lower():
                raise RuntimeError(f"Tushare 接口 {name} 权限/积分不足：{msg}") from e
            time.sleep(2 ** i)          # 多为每分钟调用上限，退避后重试
    raise last


def _to_code(ts_code: str) -> str:
    return str(ts_code).split(".")[0].zfill(6)


# ------------------------------------------------------------------ 退市股
def fetch_delisted_list() -> pd.DataFrame:
    """已退市股票列表（含退市日期）。"""
    df = _call("stock_basic", exchange="", list_status="D",
               fields="ts_code,symbol,name,list_date,delist_date,market")
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "code": df["ts_code"].map(_to_code),
        "name": df["name"].astype(str).str.replace(r"\s+", "", regex=True),
        "listed_date": pd.to_datetime(df["list_date"], errors="coerce"
                                      ).dt.strftime("%Y-%m-%d"),
        "delisted_date": pd.to_datetime(df["delist_date"], errors="coerce"
                                        ).dt.strftime("%Y-%m-%d"),
        "_ts": df["ts_code"],
    })
    out["board"] = out["code"].map(classify_board)
    out["is_st"] = 1
    out["industry"] = pd.NA
    out["float_share"] = pd.NA
    out["status"] = "delisted"
    out["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return out[out["board"] != "未知"].dropna(subset=["delisted_date"])


def fetch_daily_qfq(ts_code: str, start: str = HISTORY_START) -> pd.DataFrame:
    """单只股票的前复权日线。

    Tushare 的 daily 返回不复权价，复权因子在 adj_factor 里，需要自己合成。
    前复权 = 不复权价 × 当日因子 / 最新一日因子。对退市股而言，
    "最新"就是它最后一个交易日——锚点与在市股票不同，但收益率不受影响，
    指标和回测用的都是收益率，所以可以混用。
    """
    s = pd.to_datetime(start).strftime("%Y%m%d")
    d = _call("daily", ts_code=ts_code, start_date=s)
    if d is None or d.empty:
        return pd.DataFrame()
    f = _call("adj_factor", ts_code=ts_code, start_date=s)

    d = d.sort_values("trade_date")
    if f is not None and not f.empty:
        d = d.merge(f[["trade_date", "adj_factor"]], on="trade_date", how="left")
        d["adj_factor"] = d["adj_factor"].ffill().bfill()
        ratio = d["adj_factor"] / d["adj_factor"].iloc[-1]
    else:
        ratio = 1.0

    out = pd.DataFrame({
        "code": _to_code(ts_code),
        "date": pd.to_datetime(d["trade_date"]).dt.strftime("%Y-%m-%d"),
        "open": d["open"] * ratio,
        "high": d["high"] * ratio,
        "low": d["low"] * ratio,
        "close": d["close"] * ratio,
        "volume": d["vol"],              # 手，与 akshare 口径一致
        "amount": d["amount"] * 1000,    # 千元 -> 元
        "pct_chg": d["pct_chg"],
        "turnover": pd.NA,
    })
    return out.dropna(subset=["close"])


def sync_delisted(since: str = HISTORY_START, progress=None,
                  sleep: float = 0.35) -> dict:
    """补齐退市股的基础信息与前复权历史。

    Tushare 免费额度有每分钟调用上限，每只股票要调 2 次（daily + adj_factor），
    所以默认加了 0.35s 间隔。239 只约需 3 分钟。
    """
    store.init_db()
    lst = fetch_delisted_list()
    if lst.empty:
        return {"listed": 0, "synced": 0, "rows": 0}

    since_iso = pd.to_datetime(since).strftime("%Y-%m-%d")
    lst = lst[lst["delisted_date"] >= since_iso].reset_index(drop=True)
    store.upsert_instruments(lst.drop(columns=["_ts"]))

    have = store.last_dates()
    todo = lst[~lst["code"].isin(have)]
    stats = {"listed": len(lst), "pending": len(todo), "synced": 0,
             "failed": 0, "rows": 0}

    for i, row in enumerate(todo.itertuples(index=False), 1):
        try:
            df = fetch_daily_qfq(row._ts, since)
        except RuntimeError:
            raise                        # 权限/积分问题直接抛出，别静默跳过
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            stats["failed"] += 1
        else:
            stats["rows"] += store.upsert_daily(df)
            stats["synced"] += 1
        if progress and i % 10 == 0:
            progress(i, len(todo), stats)
        time.sleep(sleep)
    return stats
