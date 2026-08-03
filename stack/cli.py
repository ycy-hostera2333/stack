"""命令行入口：数据同步、回测、出信号。

  python -m stack.cli sync --instruments        # 更新股票列表
  python -m stack.cli sync --daily              # 增量更新全市场日线（首次很慢）
  python -m stack.cli sync --daily --limit 300  # 只同步流动性最好的 300 只，先跑通
  python -m stack.cli backtest ma_trend --start 2021-01-01
  python -m stack.cli signals ma_trend
  python -m stack.cli serve                     # 启动 Web 界面
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

from . import signals as sig
from .backtest import engine
from .data import source, store, universe
from .strategies import all_strategies, get_strategy


def _progress(done: int, total: int, stats: dict) -> None:
    pct = done / total * 100
    print(f"\r  {done}/{total} ({pct:5.1f}%)  成功 {stats['ok']}  失败 {stats['failed']}"
          f"  写入 {stats['rows']:,} 行", end="", flush=True)


def cmd_sync(args) -> None:
    store.init_db()
    t0 = time.time()

    if args.instruments or not (args.daily or args.index):
        print("同步股票列表…")
        n = source.sync_instruments()
        print(f"  股票列表 {n} 只")

    if args.index or args.daily:
        print("同步指数基准…")
        for symbol, (label, _) in source.BENCHMARKS.items():
            n = source.sync_index(symbol)
            print(f"  {label}({symbol}) {n} 根")

    if args.daily:
        codes = None
        if args.limit:
            uni = universe.build()
            if uni.empty:
                print("  股票池为空——先跑一次不带 --limit 的同步，或放宽过滤条件")
                inst = store.load_instruments()
                codes = inst["code"].head(args.limit).tolist()
            else:
                codes = uni["code"].head(args.limit).tolist()
            print(f"同步日线（限流动性前 {len(codes)} 只）…")
        else:
            print("同步全市场日线（首次约 15-30 分钟，中断后可重跑续传）…")
        stats = source.sync_daily(codes=codes, full=args.full,
                                  only_missing=args.only_missing,
                                  progress=_progress)
        print(f"\n  待更新 {stats['pending']} 只，成功 {stats['ok']}，"
              f"失败 {stats['failed']}，写入 {stats['rows']:,} 行"
              f"（共 {stats.get('passes', 1)} 轮）")
        if stats["failed"]:
            print(f"  仍有 {stats['failed']} 只未取到，多半是上游限流。"
                  f"过一会儿重跑本命令会自动只补这些。")

    cov = store.coverage()
    print(f"\n本地库：{cov['codes_with_data']} 只有数据 / {cov['instruments']} 只已登记，"
          f"{cov['bars']:,} 根 K 线，区间 {cov['first_date']} ~ {cov['last_date']}")
    print(f"耗时 {time.time() - t0:.0f}s")


def cmd_backtest(args) -> None:
    strat = get_strategy(args.strategy, **json.loads(args.params) if args.params else {})
    flt = universe.UniverseFilter(min_amount=args.min_amount)
    # 按起始日筛池子，不能用结束日——否则等于用未来的流动性决定当初买什么
    uni = universe.build(flt, as_of=args.start)
    if uni.empty:
        print("股票池为空，请先同步行情数据")
        return
    if args.top:
        uni = uni.head(args.top)

    cfg = engine.BacktestConfig(
        initial_cash=args.cash, max_positions=args.max_positions,
        stop_loss=args.stop_loss, take_profit=args.take_profit,
        trail_stop_atr=args.trail_stop_atr, max_hold_days=args.max_hold_days,
    )
    print(f"策略 {strat.label}（{strat.name}）  股票池 {len(uni)} 只  "
          f"{args.start} ~ {args.end}")
    print("回测中…")
    t0 = time.time()
    res = engine.run(strat, uni["code"].tolist(), args.start, args.end, cfg,
                     names=dict(zip(uni["code"], uni["name"])))
    print(f"完成，耗时 {time.time() - t0:.1f}s\n")
    _print_metrics(res)


def _pct(v) -> str:
    return "—" if v is None else f"{v:+.2%}"


def _print_metrics(res) -> None:
    m = res.metrics
    if "error" in m:
        print(f"错误：{m['error']}")
        return
    print("─" * 56)
    print(f"  总收益      {_pct(m['total_return']):>12}"
          f"      年化    {_pct(m['cagr']):>10}")
    print(f"  最大回撤    {_pct(m['max_drawdown']):>12}"
          f"      卡玛    {m['calmar']:>10.2f}")
    print(f"  夏普        {m['sharpe']:>12.2f}      索提诺  {m['sortino']:>10.2f}")
    print(f"  年化波动    {_pct(m['volatility']):>12}"
          f"      水下最长{m['max_drawdown_days']:>8} 日")
    if m.get("benchmark_return") is not None:
        print(f"  沪深300     {_pct(m['benchmark_return']):>12}"
              f"      超额    {_pct(m['excess_return']):>10}")
    print("─" * 56)
    if m.get("trades"):
        print(f"  交易 {m['trades']} 笔   胜率 {m['win_rate']:.1%}   "
              f"盈亏比 {m['profit_factor'] or '—'}   平均持有 {m['avg_hold_days']} 日")
        print(f"  平均每笔 {_pct(m['avg_return'])}   "
              f"盈 {_pct(m['avg_win'])} / 亏 {_pct(m['avg_loss'])}   "
              f"最好 {_pct(m['best_trade'])} / 最差 {_pct(m['worst_trade'])}")
    else:
        print(f"  {m.get('note', '无交易')}")
    if res.skipped:
        print("  受限未成交：" + "，".join(f"{k} {v} 次" for k, v in res.skipped.items()))
    print("─" * 56)


def cmd_signals(args) -> None:
    strat = get_strategy(args.strategy, **(json.loads(args.params) if args.params else {}))
    res = sig.generate(strat, cfg=sig.SignalConfig(
        portfolio_value=args.cash, max_positions=args.max_positions))
    if "error" in res:
        print(f"错误：{res['error']}")
        return
    print(f"\n{res['strategy_label']}  截止 {res['as_of']}  "
          f"股票池 {res['universe_size']} 只")
    print(f"{res['note']}\n")
    if res.get("regime_note"):
        print(f"※ {res['regime_note']}\n")

    if res["sells"]:
        print("【卖出提醒】")
        for s in res["sells"]:
            print(f"  {s['code']} {s['name']:<8} {s['price']:>8.2f}   {s['reason']}")
        print()

    if res["buys"]:
        print("【买入候选】")
        for i, b in enumerate(res["buys"], 1):
            if b.get("affordable", True):
                size = (f"建议 {b['suggest_shares']:>6} 股 / "
                        f"{b['suggest_amount']:>12,.0f} 元")
            else:
                size = f"⚠ 买不起一手（需 {b['min_amount']:,.0f} 元）"
            print(f"  {i:>2}. {b['code']} {b['name']:<8} {b['price']:>8.2f}  {size}")
            print(f"      {b['reason']}")
    else:
        print("今日无买入信号。")

    if args.save:
        n = sig.persist(res)
        print(f"\n已记录 {n} 条信号到本地库")


def cmd_list(args) -> None:
    print("可用策略：\n")
    for s in all_strategies():
        print(f"  {s['name']:<20} {s['label']}")
        print(f"    {s['description']}")
        print(f"    默认参数：{s['defaults']}\n")


def cmd_paper(args) -> None:
    from . import paper

    if args.action == "init":
        params = json.loads(args.params) if args.params else {}
        paper.reset(args.strategy, params, args.cash, args.max_positions, args.top)
        print(f"模拟盘已建立：{args.strategy}  本金 {args.cash:,.0f}  "
              f"{args.max_positions} 仓位  股票池前 {args.top} 只")
        if args.since:
            print(f"回补 {args.since} 至今…")
            # 必须按日期升序逐日推进：advance 会把 last_date 前移，
            # 一旦先处理了最新日期，之前的日期都会被「已处理过」的守卫挡掉
            done = 0
            for d in store.trading_days(start=args.since):
                ev = paper.advance(as_of=d, verbose=False)
                if not ev.get("skipped"):
                    done += 1
            print(f"  已回补 {done} 个交易日")
        return

    if args.action == "run":
        n = paper.catch_up()
        print(f"推进了 {n} 个交易日" if n else "已是最新，无需推进")

    st = paper.status()
    if not st.get("strategy"):
        print("模拟盘尚未初始化，先运行：paper init")
        return

    print(f"\n{'─'*60}")
    print(f"  模拟盘 · {st['strategy']}   建于 {st['created_at']}")
    print(f"{'─'*60}")
    print(f"  本金 {st['initial_cash']:>12,.0f}      运行 {st['days']} 个交易日")
    if "equity" in st:
        print(f"  净值 {st['equity']:>12,.0f}      收益 {st['total_return']:+.2%}"
              f"      回撤 {st['max_drawdown']:.2%}")
        if st.get("benchmark_return") is not None:
            print(f"  沪深300 {st['benchmark_return']:+.2%}"
                  f"      超额 {st['total_return']-st['benchmark_return']:+.2%}")
        print(f"  现金 {st['cash']:>12,.0f}      择时空仓 {st.get('idle_days',0)} 天")
    print(f"  已平仓 {st['closed_trades']} 笔"
          + (f"   胜率 {st['win_rate']:.0%}   已实现 {st['realized_pnl']:+,.0f}"
             if st["closed_trades"] else ""))
    if st["holdings"]:
        print(f"\n  当前持仓 {len(st['holdings'])} 只：")
        for h in st["holdings"]:
            print(f"    {h['code']} {h['name']:<8} {h['shares']:>6}股  "
                  f"成本 {h['cost']:>8.2f}  买于 {h['open_date']}  "
                  f"持有 {h['hold_days']}日")
    print(f"{'─'*60}")
    print(f"  每天收盘后跑一次 `paper run`。这些记录一旦写下就不再改动——")
    print(f"  半年后回看时，它才是真正没被调过参的未来数据。")


def cmd_selftest(args) -> None:
    from .selftest import run_all
    print("自检：验证回测引擎的关键不变量…")
    ok = run_all()
    sys.exit(0 if ok else 1)


def cmd_serve(args) -> None:
    import uvicorn
    print(f"界面地址： http://127.0.0.1:{args.port}")
    uvicorn.run("stack.api:app", host="127.0.0.1", port=args.port, reload=args.reload)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="stack", description="A股选股与信号辅助系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="同步行情数据")
    s.add_argument("--instruments", action="store_true", help="更新股票列表")
    s.add_argument("--daily", action="store_true", help="更新日线")
    s.add_argument("--index", action="store_true", help="更新指数基准")
    s.add_argument("--full", action="store_true", help="重建全部历史（复权价会变，建议定期跑）")
    s.add_argument("--only-missing", action="store_true",
                   help="只补从没取到过的股票，已有数据的跳过。上游限流后专门补缺用")
    s.add_argument("--limit", type=int, help="只同步流动性最好的前 N 只")
    s.set_defaults(func=cmd_sync)

    today = datetime.now().strftime("%Y-%m-%d")
    b = sub.add_parser("backtest", help="回测策略")
    b.add_argument("strategy")
    b.add_argument("--start", default="2021-01-01")
    b.add_argument("--end", default=today)
    b.add_argument("--cash", type=float, default=200_000)
    b.add_argument("--max-positions", type=int, default=5)
    b.add_argument("--stop-loss", type=float, default=0.0, help="如 0.08 表示 -8% 止损")
    b.add_argument("--take-profit", type=float, default=0.0)
    b.add_argument("--trail-stop-atr", type=float, default=0.0,
                   help="跟踪止损倍数，如 2.5 表示自持仓最高价回落 2.5×ATR 卖出")
    b.add_argument("--max-hold-days", type=int, default=0)
    b.add_argument("--min-amount", type=float, default=5e7, help="20日均成交额下限")
    b.add_argument("--top", type=int, help="只用流动性前 N 只做回测")
    b.add_argument("--params", help='策略参数 JSON，如 \'{"atr_stop":3}\'')
    b.set_defaults(func=cmd_backtest)

    g = sub.add_parser("signals", help="生成今日信号")
    g.add_argument("strategy")
    g.add_argument("--cash", type=float, default=200_000)
    g.add_argument("--max-positions", type=int, default=5)
    g.add_argument("--save", action="store_true", help="记录到本地库")
    g.add_argument("--params", help='策略参数 JSON，如 \'{"fast":5,"slow":20}\'')
    g.set_defaults(func=cmd_signals)

    sub.add_parser("list", help="列出可用策略").set_defaults(func=cmd_list)
    sub.add_parser("stats", help="本地行情库统计画像").set_defaults(
        func=lambda a: __import__("stack.stats", fromlist=["x"]).report(
            __import__("stack.stats", fromlist=["x"]).collect()))
    sub.add_parser("selftest", help="自检：验证回测引擎的关键不变量"
                   ).set_defaults(func=cmd_selftest)

    pp = sub.add_parser("paper", help="前向模拟盘：逐日推进的虚拟账户")
    pp.add_argument("action", choices=["init", "run", "status"],
                    help="init=新建(清空重来)  run=推进到最新  status=只看状态")
    pp.add_argument("--strategy", default="regime_momentum")
    pp.add_argument("--params", help='策略参数 JSON')
    pp.add_argument("--cash", type=float, default=200_000)
    pp.add_argument("--max-positions", type=int, default=5)
    pp.add_argument("--top", type=int, default=400, help="股票池取流动性前 N 只")
    pp.add_argument("--since", help="init 时从该日期开始回补，如 2026-06-01")
    pp.set_defaults(func=cmd_paper)

    v = sub.add_parser("serve", help="启动 Web 界面")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--reload", action="store_true")
    v.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
