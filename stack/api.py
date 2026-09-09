"""FastAPI 后端。

只监听 127.0.0.1，纯自用，不做鉴权。若要放到局域网/公网，请自行加认证。
回测是 CPU 密集型任务，跑在线程池里，避免阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import WEB_DIR
from . import signals as sig
from .backtest import engine
from .data import source, store, universe
from .strategies import all_strategies, get_strategy

@asynccontextmanager
async def _lifespan(app: FastAPI):
    _start_paper_daemon()
    yield


app = FastAPI(title="Stack · A股选股与信号系统", docs_url="/api/docs",
              lifespan=_lifespan)
store.init_db()


def _clean(obj):
    """把 NaN/NaT 换成 None，否则 JSON 序列化会产出非法的 NaN 字面量。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


async def _run(fn, *a, **kw):
    return await asyncio.get_running_loop().run_in_executor(None, partial(fn, *a, **kw))


# ------------------------------------------------------------------ 请求模型
class UniverseReq(BaseModel):
    exclude_st: bool = True
    exclude_star: bool = False
    exclude_chinext: bool = False
    exclude_bj: bool = True
    min_listed_days: int = 250
    min_amount: float = 5e7
    min_price: float = 2.0

    def to_filter(self) -> universe.UniverseFilter:
        return universe.UniverseFilter(**self.model_dump())


class BacktestReq(BaseModel):
    strategy: str
    start: str = "2021-01-01"
    # 默认到库内最后一个完整交易日，而不是今天——库尾残日会让最后几天冻住
    end: str = Field(default_factory=lambda: (
        store.last_complete_day() or datetime.now().strftime("%Y-%m-%d")))
    initial_cash: float = 200_000
    max_positions: int = 5
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trail_stop_atr: float = 0.0
    max_hold_days: int = 0
    top: int = 800
    params: dict = Field(default_factory=dict)
    universe: UniverseReq = Field(default_factory=UniverseReq)


class SignalReq(BaseModel):
    strategy: str
    portfolio_value: float = 200_000
    max_positions: int = 5
    max_candidates: int = 15
    save: bool = False
    allow_partial_bar: bool = False
    params: dict = Field(default_factory=dict)
    universe: UniverseReq = Field(default_factory=UniverseReq)


class PositionReq(BaseModel):
    code: str
    name: str = ""
    shares: int
    cost: float
    open_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    note: str = ""


class SimulatorSaveReq(BaseModel):
    user_name: str = Field(min_length=1, max_length=40)
    state: dict


class PaperAccountReq(BaseModel):
    account: str = Field(min_length=1, max_length=32)
    strategy: str
    params: dict = Field(default_factory=dict)
    cash: float = 200_000
    max_positions: int = 5
    top: int = 400
    max_hold_days: int = 0
    since: str = ""          # 从该日回补，空则从今天开始记


class PaperDaemonReq(BaseModel):
    enabled: bool | None = None
    auto_sync: bool | None = None


# ------------------------------------------------------------------ 基础信息
@app.get("/api/status")
async def status():
    cov = await _run(store.coverage)
    return {
        **cov,
        "last_complete_day": await _run(store.last_complete_day),
        "instruments_synced_at": store.get_meta("instruments_synced_at"),
        "daily_synced_at": store.get_meta("daily_synced_at"),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/strategies")
async def strategies():
    return all_strategies()


@app.post("/api/universe")
async def get_universe(req: UniverseReq):
    uni = await _run(universe.build, req.to_filter())
    if uni.empty:
        return {"count": 0, "by_board": {}, "rows": []}
    show = uni.head(300)[["code", "name", "board", "industry",
                          "avg_amount", "last_close"]]
    return _clean({
        "count": int(len(uni)),
        "by_board": uni["board"].value_counts().to_dict(),
        "rows": show.to_dict("records"),
    })


# ------------------------------------------------------------------ 历史行情回放
@app.get("/api/replay/instruments")
async def replay_instruments(q: str = "", limit: int = 80):
    """供模拟器选择股票；仅返回本地确实已有日线的标的。"""
    out = await _run(store.instruments_with_data, q, max(1, min(limit, 200)))
    return _clean(out.head(max(1, min(limit, 200))).to_dict("records"))


@app.get("/api/replay/bars")
async def replay_bars(code: str, start: str | None = None, end: str | None = None):
    code = str(code).strip().zfill(6)
    bars = await _run(store.load_daily, [code], start, end)
    if bars.empty:
        raise HTTPException(404, "该股票在所选时段没有本地日线数据")
    inst = await _run(store.load_instruments)
    hit = inst[inst["code"].astype(str) == code] if not inst.empty else pd.DataFrame()
    name = str(hit["name"].iloc[0]) if not hit.empty else code
    cols = ["date", "open", "high", "low", "close", "volume"]
    bars["date"] = bars["date"].dt.strftime("%Y-%m-%d")
    return _clean({"code": code, "name": name, "bars": bars[cols].to_dict("records")})


@app.get("/api/replay/saves")
async def replay_saves():
    return await _run(store.list_simulator_saves)


@app.post("/api/replay/saves")
async def save_replay(req: SimulatorSaveReq):
    name = req.user_name.strip()
    if not name:
        raise HTTPException(400, "请输入存档用户名")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await _run(store.save_simulator, name, json.dumps(_clean(req.state), ensure_ascii=False), now)
    return {"ok": True, "updated_at": now}


@app.get("/api/replay/saves/{user_name}")
async def load_replay(user_name: str):
    saved = await _run(store.load_simulator, user_name)
    if not saved:
        raise HTTPException(404, "没有找到该用户的存档")
    return {"state": json.loads(saved["payload"]), "updated_at": saved["updated_at"]}


# ------------------------------------------------------------------ 回测
@app.post("/api/backtest")
async def backtest(req: BacktestReq):
    try:
        strat = get_strategy(req.strategy, **req.params)
    except KeyError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        # 参数非法（如快线周期 >= 慢线周期），把原因原样告诉界面
        raise HTTPException(400, str(e))

    # 股票池按回测**起始日**的流动性和价格筛选，不能用 req.end：
    # 用结束日的数据选股等于拿未来信息决定当初买什么，是典型的前视偏差。
    uni = await _run(universe.build, req.universe.to_filter(), req.start)
    if uni.empty:
        raise HTTPException(400, "股票池为空，请先同步行情或放宽过滤条件")
    if req.top:
        uni = uni.head(req.top)

    cfg = engine.BacktestConfig(
        initial_cash=req.initial_cash, max_positions=req.max_positions,
        stop_loss=req.stop_loss, take_profit=req.take_profit,
        trail_stop_atr=req.trail_stop_atr, max_hold_days=req.max_hold_days,
    )
    res = await _run(engine.run, strat, uni["code"].tolist(), req.start, req.end,
                     cfg, dict(zip(uni["code"], uni["name"])))
    payload = res.to_json()
    payload["universe_size"] = int(len(uni))
    return JSONResponse(_clean(payload))


# ------------------------------------------------------------------ 信号
@app.post("/api/signals")
async def get_signals(req: SignalReq):
    try:
        strat = get_strategy(req.strategy, **req.params)
    except KeyError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        # 参数非法（如快线周期 >= 慢线周期），把原因原样告诉界面
        raise HTTPException(400, str(e))
    cfg = sig.SignalConfig(max_candidates=req.max_candidates,
                           portfolio_value=req.portfolio_value,
                           max_positions=req.max_positions,
                           allow_partial_bar=req.allow_partial_bar)
    res = await _run(sig.generate, strat, req.universe.to_filter(), cfg)
    if req.save and "error" not in res:
        res["saved"] = await _run(sig.persist, res)
    return JSONResponse(_clean(res))


@app.get("/api/signals/history")
async def signal_history(limit: int = 200):
    df = await _run(store.load_signal_log, limit)
    return _clean(df.to_dict("records") if not df.empty else [])


# ------------------------------------------------------------------ 持仓
@app.get("/api/positions")
async def positions():
    df = await _run(store.list_positions)
    if df.empty:
        return {"rows": [], "total_cost": 0, "total_value": 0, "total_pnl": 0}

    codes = df["code"].astype(str).tolist()
    spot = await _run(source.fetch_spot, codes)
    price_map = dict(zip(spot["code"], spot["price"])) if not spot.empty else {}
    src_map = {c: "live" for c in price_map}

    # 取不到就退回本地最近收盘价。注意本地存的是前复权价，除权后会与实际成交价
    # 有系统性偏差，据此算出的盈亏仅供参考——所以界面上要标明价格来源。
    missing = [c for c in codes if c not in price_map or price_map[c] != price_map[c]]
    if missing:
        last = await _run(store.load_daily, missing, None, None)
        if not last.empty:
            for code, g in last.groupby("code"):
                price_map[code] = float(g.sort_values("date")["close"].iloc[-1])
                src_map[code] = "cache"

    rows, total_cost, total_value = [], 0.0, 0.0
    for r in df.to_dict("records"):
        code = str(r["code"])
        px = price_map.get(code)
        cost_amt = r["shares"] * r["cost"]
        val = r["shares"] * px if px else cost_amt
        total_cost += cost_amt
        total_value += val
        rows.append({**r, "price": px,
                     "price_source": src_map.get(code, "none"),
                     "market_value": round(val, 2),
                     "pnl": round(val - cost_amt, 2),
                     "pnl_pct": round(val / cost_amt - 1, 4) if cost_amt else 0})
    return _clean({
        "rows": rows,
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_value - total_cost, 2),
        "total_pnl_pct": round(total_value / total_cost - 1, 4) if total_cost else 0,
    })


@app.post("/api/positions")
async def add_position(req: PositionReq):
    name = req.name
    if not name:
        inst = await _run(store.load_instruments)
        hit = inst[inst["code"] == req.code]
        name = hit["name"].iloc[0] if not hit.empty else req.code
    pid = await _run(store.add_position, req.code, name, req.shares,
                     req.cost, req.open_date, req.note)
    return {"id": pid}


@app.delete("/api/positions/{pos_id}")
async def remove_position(pos_id: int):
    await _run(store.delete_position, pos_id)
    return {"ok": True}


# ------------------------------------------------------------------ 数据同步
# 线程安全内存日志：环形缓冲，同步过程实时记录每只股票来自哪个源、取了多少行。
import collections
_sync_state: dict = {"running": False, "phase": "", "done": 0,
                     "total": 0, "stats": {}, "finished_at": None,
                     "updated_at": None, "cancelled": False,
                     "circuit_breaker": 3}
_sync_log: collections.deque = collections.deque(maxlen=2000)
source_stats: dict = {"tencent": 0, "baostock": 0, "tushare": 0, "akshare": 0, "none": 0}


def _do_sync(full: bool, limit: int | None, only_missing: bool = False,
             circuit_breaker: int = 3, slow: bool = False) -> None:
    _sync_log.clear()
    for k in source_stats:
        source_stats[k] = 0
    _sync_state["cancelled"] = False
    _sync_state["circuit_breaker"] = circuit_breaker
    try:
        _sync_state.update(running=True, phase="股票列表", done=0, total=0,
                           finished_at=None, updated_at=datetime.now().strftime("%H:%M:%S"))
        source.sync_instruments()
        _sync_state["phase"] = "指数基准"
        for symbol in source.BENCHMARKS:
            source.sync_index(symbol)
        _sync_state["phase"] = "日线行情"

        codes = None
        if limit:
            uni = universe.build()
            codes = (uni["code"].head(limit).tolist() if not uni.empty
                     else store.load_instruments()["code"].head(limit).tolist())

        def prog(done, total, stats):
            _sync_state.update(done=done, total=total, stats=dict(stats),
                               updated_at=datetime.now().strftime("%H:%M:%S"))

        def _on_event(code, src, rows):
            # 记录日志 + 累加来源统计
            _sync_log.append({
                "t": datetime.now().strftime("%H:%M:%S"),
                "code": code,
                "src": src,
                "rows": rows,
            })
            if src in source_stats:
                source_stats[src] += 1

        def _cancel_check():
            return _sync_state.get("cancelled", False)

        stats = source.sync_daily(codes=codes, full=full, progress=prog,
                                  only_missing=only_missing, on_event=_on_event,
                                  circuit_breaker=circuit_breaker,
                                  cancel_check=_cancel_check, slow=slow)
        _sync_state.update(stats=stats, done=stats["pending"], total=stats["pending"],
                           updated_at=datetime.now().strftime("%H:%M:%S"))
        _sync_state["source_stats"] = dict(source_stats)
        if _sync_state["cancelled"]:
            _sync_state["phase"] = "已取消"
        elif stats.get("failed"):
            _sync_state["phase"] = f"完成，{stats['failed']} 只失败"
        else:
            _sync_state["phase"] = "完成"
    except Exception as e:
        _sync_state["phase"] = f"失败：{e}"
    finally:
        _sync_state["running"] = False
        _sync_state["source_stats"] = dict(source_stats)
        _sync_state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.post("/api/sync")
async def start_sync(full: bool = False, limit: int | None = None,
                     only_missing: bool = False, circuit_breaker: int = 3):
    if _sync_state["running"]:
        return {"started": False, "message": "同步已在进行中"}
    asyncio.get_running_loop().run_in_executor(
        None, _do_sync, full, limit, only_missing, circuit_breaker)
    return {"started": True}


@app.post("/api/sync/cancel")
async def cancel_sync():
    """请求中断当前同步。同步线程会在下一个检查点停止。"""
    _sync_state["cancelled"] = True
    return {"ok": True, "message": "取消信号已发送，同步将在近期停止"}


@app.get("/api/sync/status")
async def sync_status():
    return _sync_state


@app.get("/api/sync/errors")
async def sync_errors(limit: int = 200):
    """最近一次同步失败的具体股票与原因。"""
    df = await _run(store.load_sync_errors, max(1, min(limit, 1000)))
    return _clean(df.to_dict("records") if not df.empty else [])


@app.get("/api/sync/log")
async def sync_log():
    """实时同步日志流（每只股票来自哪个源、取了多少行）+ 各源统计。"""
    return {"log": list(_sync_log), "source_stats": dict(source_stats)}


@app.get("/api/data/gaps")
async def data_gaps():
    """数据缺失检查：找滞后股票和没有日线的股票。"""
    return await _run(store.find_gaps)


# ------------------------------------------------------------------ 模拟盘
# 前向验证的唯一失败模式是断链：没人每天去跑 paper run，记录就悄悄停在某一天，
# 半年后才发现只攒了三天数据。所以推进这件事不能靠人记得。

PAPER_TICK_SECONDS = 30 * 60
PAPER_FIRST_DELAY = 20            # 让服务先起来，别和启动抢资源

_paper_daemon: dict = {
    "enabled": True, "auto_sync": True, "running": False,
    "phase": "未启动", "last_run": None, "error": None,
    "advanced": {}, "data_lag": None, "creating": None,
    "sync_backoff": 0, "next_sync_in": None, "sync_note": None,
}
_next_sync_at = 0.0                # 单调时钟上的下次允许同步时间


def _paper_flag(key: str, default: bool = True) -> bool:
    v = store.get_meta(f"paper_daemon_{key}")
    return default if v is None else v == "1"


def _source_alive() -> bool:
    """轻量探活：走一遍 fallback 链抓一只票最近几天。

    限流通常几十分钟就过去了，而退避最长会等到 8 小时——数据源早恢复了还在干等，
    白白丢掉几天前向记录。一次请求就能知道该不该提前解除。
    """
    from datetime import timedelta

    from .data import source

    end = datetime.now()
    start = (end - timedelta(days=12)).strftime("%Y-%m-%d")
    try:
        df = source.fetch_daily("600000", start=start,
                                end=end.strftime("%Y-%m-%d"))
        return df is not None and not df.empty
    except Exception:
        return False


def _paper_tick() -> None:
    """一次自愈：数据落后就同步，然后把每个账户推进到最新。"""
    from . import paper

    _paper_daemon["enabled"] = _paper_flag("enabled")
    _paper_daemon["auto_sync"] = _paper_flag("auto_sync")
    if not _paper_daemon["enabled"]:
        _paper_daemon["phase"] = "已停用"
        return

    _paper_daemon.update(running=True, error=None, phase="检查数据")
    try:
        global _next_sync_at
        stale = paper.data_lag_days()
        _paper_daemon["data_lag"] = stale
        now = time.monotonic()
        if stale and _paper_daemon["auto_sync"]:
            if now < _next_sync_at and not _source_alive():
                wait = int((_next_sync_at - now) / 60)
                _paper_daemon["next_sync_in"] = wait
                _paper_daemon["phase"] = (f"行情落后 {stale} 天，"
                                          f"数据源仍不通，{wait} 分钟后再试")
            elif _sync_state["running"]:
                _paper_daemon["phase"] = "手动同步进行中，本轮跳过同步"
            else:
                if _next_sync_at:      # 探活把退避提前解除了，记一笔
                    _paper_daemon["sync_note"] = "数据源已恢复，提前结束退避重试"
                    _next_sync_at = 0.0
                # 已经退避过一轮，说明上游在限流——这时候再用常规档
                # （33 请求/秒）去撞，几百只就又被封。改走慢速档。
                slow = _paper_daemon.get("sync_backoff", 0) > 0
                _paper_daemon["phase"] = (f"行情落后 {stale} 天，正在同步"
                                          + ("（慢速档）" if slow else ""))
                _do_sync(False, None, False, 3, slow)
                after = paper.data_lag_days()
                _paper_daemon["data_lag"] = after
                if after < stale:
                    _paper_daemon.update(sync_backoff=0, next_sync_in=None,
                                         sync_note=None)
                    _next_sync_at = 0.0
                else:
                    # 同步跑完了，行情一天都没往前挪——多半是数据源在限流或全挂。
                    # 每 30 分钟拿几千次注定失败的请求去砸它们，既没用也不礼貌，
                    # 还会让限流更久。指数退避，最长 8 小时。
                    b = min(_paper_daemon.get("sync_backoff", 0) + 1, 4)
                    failed = (_sync_state.get("stats") or {}).get("failed")
                    _paper_daemon["sync_backoff"] = b
                    _next_sync_at = time.monotonic() + PAPER_TICK_SECONDS * (2 ** b)
                    _paper_daemon["next_sync_in"] = int(
                        PAPER_TICK_SECONDS * (2 ** b) / 60)
                    _paper_daemon["sync_note"] = (
                        f"同步跑完了但行情仍停在库内最后一个完整交易日"
                        + (f"，{failed} 只全部数据源都失败" if failed else "")
                        + "。多半是数据源限流，等一等再说；"
                        "急的话到「数据管理」手动重试。")

        _paper_daemon["phase"] = "推进模拟盘"
        adv = {}
        for a in paper.list_accounts():
            name = a["account"]
            try:
                adv[name] = paper.catch_up(verbose=False, account=name)
            except Exception as e:
                # 一个账户炸了不能拖垮其他账户，但也绝不能吞掉——记下来给界面看
                adv[name] = f"失败：{type(e).__name__}: {e}"
        _paper_daemon["advanced"] = adv
        bad = [k for k, v in adv.items() if isinstance(v, str)]
        _paper_daemon["phase"] = (f"完成，{len(bad)} 个账户出错" if bad else "完成")
    finally:
        _paper_daemon["running"] = False
        _paper_daemon["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _paper_loop() -> None:
    time.sleep(PAPER_FIRST_DELAY)
    while True:
        try:
            _paper_tick()
        except Exception as e:
            _paper_daemon.update(
                running=False, phase="失败",
                error=f"{type(e).__name__}: {e}",
                last_run=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(PAPER_TICK_SECONDS)


_decay_running: set = set()
_paper_thread: threading.Thread | None = None


def _start_paper_daemon() -> None:
    global _paper_thread
    if _paper_thread and _paper_thread.is_alive():
        return
    _paper_thread = threading.Thread(target=_paper_loop, daemon=True,
                                     name="paper-daemon")
    _paper_thread.start()


def _do_create_account(req: PaperAccountReq) -> None:
    """新建账户并回补。回补要逐日推进，几十个交易日就是几十秒，所以走后台。"""
    from . import paper

    try:
        _paper_daemon["creating"] = {"account": req.account, "phase": "建立账户",
                                     "done": 0, "total": 0}
        paper.reset(req.strategy, req.params, req.cash, req.max_positions,
                    req.top, req.max_hold_days, account=req.account)
        if req.since:
            days = store.trading_days(start=req.since)
            _paper_daemon["creating"].update(phase="回补", total=len(days))
            done = 0
            # 必须按日期升序逐日推进：advance 会把 last_date 前移，
            # 一旦先处理了最新日期，之前的日期都会被「已处理过」的守卫挡掉
            for d in days:
                ev = paper.advance(as_of=d, verbose=False, account=req.account)
                if ev.get("skipped"):
                    continue
                done += 1
                _paper_daemon["creating"]["done"] = done
        _paper_daemon["creating"].update(phase="完成")
    except Exception as e:
        _paper_daemon["creating"] = {"account": req.account, "phase": "失败",
                                     "error": f"{type(e).__name__}: {e}"}


@app.get("/api/paper/accounts")
async def paper_accounts():
    from . import paper

    def _load():
        rows = paper.list_accounts()
        for r in rows:
            r["lag_days"] = paper.lag_days(r["account"])
        return {"accounts": rows, "data_lag_days": paper.data_lag_days()}

    return _clean(await _run(_load))


@app.get("/api/paper/status")
async def paper_status(account: str = "default"):
    from . import paper
    return _clean(await _run(paper.status, account))


@app.get("/api/paper/trades")
async def paper_trades(account: str = "default", limit: int = 200):
    from . import paper
    return _clean(await _run(paper.trades, account, max(1, min(limit, 1000))))


@app.post("/api/paper/accounts")
async def paper_create(req: PaperAccountReq):
    from . import paper

    try:
        paper.check_name(req.account)
        get_strategy(req.strategy, **req.params)       # 参数不合法就别建了
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    cur = _paper_daemon.get("creating")
    if cur and cur.get("phase") not in ("完成", "失败", None):
        raise HTTPException(409, f"正在建立账户 {cur['account']}，请稍候")
    if (await _run(paper.status, req.account)).get("strategy"):
        raise HTTPException(409, f"账户 {req.account} 已存在。"
                                 "重置会清空它已积累的前向记录，请先删除再建。")
    asyncio.get_running_loop().run_in_executor(None, _do_create_account, req)
    return {"started": True, "account": req.account}


@app.delete("/api/paper/accounts/{account}")
async def paper_drop(account: str):
    from . import paper

    try:
        paper.check_name(account)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return await _run(paper.drop_account, account)


@app.get("/api/paper/decay")
async def paper_decay(account: str = "default", refresh: bool = False):
    """前向表现 vs 该策略历史分布。要跑一次历史回测，所以结果按天缓存。"""
    from . import paper

    if account in _decay_running:
        raise HTTPException(409, "该账户的衰减报告正在计算中")
    _decay_running.add(account)
    try:
        return _clean(await _run(paper.decay_report, account, 20, 4, refresh))
    finally:
        _decay_running.discard(account)


@app.get("/api/paper/daemon")
async def paper_daemon_status():
    _paper_daemon["enabled"] = _paper_flag("enabled")
    _paper_daemon["auto_sync"] = _paper_flag("auto_sync")
    _paper_daemon["alive"] = bool(_paper_thread and _paper_thread.is_alive())
    _paper_daemon["interval_seconds"] = PAPER_TICK_SECONDS
    return _clean(_paper_daemon)


@app.post("/api/paper/daemon")
async def paper_daemon_set(req: PaperDaemonReq):
    if req.enabled is not None:
        store.set_meta("paper_daemon_enabled", "1" if req.enabled else "0")
        _paper_daemon["enabled"] = req.enabled
    if req.auto_sync is not None:
        store.set_meta("paper_daemon_auto_sync", "1" if req.auto_sync else "0")
        _paper_daemon["auto_sync"] = req.auto_sync
    return {"enabled": _paper_daemon["enabled"],
            "auto_sync": _paper_daemon["auto_sync"]}


# ------------------------------------------------------------------ 前端
@app.get("/")
@app.head("/")     # 不加 HEAD，健康检查/探活工具会收到 405
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
async def favicon():
    # 内联一个极小的 SVG，省掉一次 404，也不用额外的静态文件
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
           '<rect width="16" height="16" rx="3" fill="#4c9aff"/>'
           '<path d="M3 11l3-4 3 2 4-6" stroke="#fff" stroke-width="1.8" '
           'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "max-age=86400"})


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
