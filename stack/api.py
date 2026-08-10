"""FastAPI 后端。

只监听 127.0.0.1，纯自用，不做鉴权。若要放到局域网/公网，请自行加认证。
回测是 CPU 密集型任务，跑在线程池里，避免阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import json
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

app = FastAPI(title="Stack · A股选股与信号系统", docs_url="/api/docs")
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
    end: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
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


# ------------------------------------------------------------------ 基础信息
@app.get("/api/status")
async def status():
    cov = await _run(store.coverage)
    return {
        **cov,
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
             circuit_breaker: int = 3) -> None:
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
                                  cancel_check=_cancel_check)
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
