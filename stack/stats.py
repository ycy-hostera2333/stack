"""本地行情库的统计画像。

重活交给 SQL 聚合，不把 800 万行拉进内存。

这里挑的指标都是**会改变策略设计**的那类，而不是好看的摘要：
收益分布的偏度决定要不要止盈、涨跌停频率决定信号能不能执行、
停牌频率决定回测里"卖不掉"有多常见、市场宽度决定择时值不值得做。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import store


def _sql(conn, q, params=()):
    return conn.execute(q, params).fetchall()


def collect() -> dict:
    out: dict = {}
    with store.connect() as c:
        # ---------------- 覆盖度 ----------------
        row = _sql(c, """SELECT COUNT(*), COUNT(DISTINCT code), MIN(date), MAX(date)
                         FROM daily WHERE code NOT LIKE 'IDX%'""")[0]
        out["cover"] = {"bars": row[0], "codes": row[1],
                        "first": row[2], "last": row[3]}
        out["instruments"] = _sql(c, "SELECT COUNT(*) FROM instruments")[0][0]

        # ---------------- 市场构成 ----------------
        out["by_board"] = dict(_sql(c, """SELECT board, COUNT(*) FROM instruments
                                          GROUP BY board ORDER BY COUNT(*) DESC"""))
        out["st_count"] = _sql(c, "SELECT COUNT(*) FROM instruments WHERE is_st=1")[0][0]
        out["by_listed_year"] = dict(_sql(c, """
            SELECT substr(listed_date,1,4) y, COUNT(*) FROM instruments
            WHERE listed_date IS NOT NULL GROUP BY y ORDER BY y DESC LIMIT 8"""))

        # ---------------- 每年交易日与个股覆盖 ----------------
        out["by_year"] = _sql(c, """
            SELECT substr(date,1,4) y, COUNT(DISTINCT date), COUNT(DISTINCT code),
                   COUNT(*) FROM daily WHERE code NOT LIKE 'IDX%'
            GROUP BY y ORDER BY y""")

        # ---------------- 涨跌停频率 ----------------
        # 用 pct_chg 近似判定：主板 ±9.8% 以上、20cm 板 ±19.5% 以上
        out["limit_stats"] = _sql(c, """
            SELECT substr(date,1,4) y,
              SUM(CASE WHEN (substr(code,1,3) IN ('300','301','688','689')
                             AND pct_chg >= 19.5)
                        OR (substr(code,1,3) NOT IN ('300','301','688','689')
                             AND pct_chg >= 9.8) THEN 1 ELSE 0 END),
              SUM(CASE WHEN (substr(code,1,3) IN ('300','301','688','689')
                             AND pct_chg <= -19.5)
                        OR (substr(code,1,3) NOT IN ('300','301','688','689')
                             AND pct_chg <= -9.8) THEN 1 ELSE 0 END),
              COUNT(*)
            FROM daily WHERE code NOT LIKE 'IDX%' AND pct_chg IS NOT NULL
            GROUP BY y ORDER BY y""")

        # ---------------- 指数分年度 ----------------
        idx = {}
        for code, name in [("IDX000300", "沪深300"), ("IDX000905", "中证500"),
                           ("IDX000001", "上证指数"), ("IDX399006", "创业板指")]:
            rows = _sql(c, """SELECT date, close FROM daily WHERE code=?
                              ORDER BY date""", (code,))
            if not rows:
                continue
            s = pd.Series([r[1] for r in rows],
                          index=pd.to_datetime([r[0] for r in rows]))
            yr = s.resample("YE").last()
            first = s.resample("YE").first()
            idx[name] = {str(t.year): round(float(yr.iloc[i] / first.iloc[i] - 1), 4)
                         for i, t in enumerate(yr.index)}
        out["indices"] = idx

    # ---------------- 个股年度收益分布 ----------------
    # 逐年取首末收盘价算涨跌幅，看分布形态（均值 vs 中位数 = 右偏程度）
    with store.connect() as c:
        yearly = pd.read_sql("""
            SELECT substr(date,1,4) y, code, date, close FROM daily
            WHERE code NOT LIKE 'IDX%' AND close > 0""", c)
    dist = []
    for y, g in yearly.groupby("y"):
        g = g.sort_values("date")
        agg = g.groupby("code")["close"].agg(["first", "last", "size"])
        agg = agg[agg["size"] >= 100]          # 全年至少 100 个交易日
        if agg.empty:
            continue
        r = (agg["last"] / agg["first"] - 1).to_numpy()
        dist.append({
            "year": y, "n": len(r),
            "mean": float(r.mean()), "median": float(np.median(r)),
            "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90)),
            "up": int((r > 0).sum()), "down": int((r <= 0).sum()),
            "gt100": int((r > 1.0).sum()), "lt50": int((r < -0.5).sum()),
            "max": float(r.max()),
        })
    out["yearly_dist"] = dist

    # ---------------- 流动性与波动率分位 ----------------
    last = out["cover"]["last"]
    if last:
        start = (pd.Timestamp(last) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        with store.connect() as c:
            recent = pd.read_sql("""
                SELECT code, amount, close, pct_chg FROM daily
                WHERE code NOT LIKE 'IDX%' AND date >= ?""", c, params=[start])
        if not recent.empty:
            amt = recent.groupby("code")["amount"].mean().dropna()
            vol = (recent.groupby("code")["pct_chg"].std() * np.sqrt(252) / 100).dropna()
            px = recent.groupby("code")["close"].last().dropna()
            out["liquidity"] = {f"p{p}": float(np.percentile(amt, p))
                                for p in (10, 25, 50, 75, 90)}
            out["volatility"] = {f"p{p}": float(np.percentile(vol, p))
                                 for p in (10, 25, 50, 75, 90)}
            out["price"] = {f"p{p}": float(np.percentile(px, p))
                            for p in (10, 25, 50, 75, 90, 99)}
            out["price_over400"] = int((px > 400).sum())

    # ---------------- 停牌与数据完整度 ----------------
    days = store.trading_days()
    out["trading_days"] = len(days)
    with store.connect() as c:
        per = pd.read_sql("""SELECT code, COUNT(*) n, MIN(date) f, MAX(date) l
                             FROM daily WHERE code NOT LIKE 'IDX%' GROUP BY code""", c)
    if not per.empty and days:
        # 只对全程在市的股票算停牌：上市日早于库起点、且最后一天是最新交易日
        full = per[(per["f"] <= days[0]) & (per["l"] == days[-1])]
        out["full_history_codes"] = int(len(full))
        if len(full):
            miss = (len(days) - full["n"]) / len(days)
            out["suspend"] = {"mean": float(miss.mean()),
                              "median": float(np.median(miss)),
                              "p90": float(np.percentile(miss, 90)),
                              "over10pct": int((miss > 0.10).sum())}
        # 已退市/停更：最后一根 K 线远早于最新交易日
        stale = per[per["l"] < days[-20]] if len(days) > 20 else per.iloc[:0]
        out["stale_codes"] = int(len(stale))
    return out


# ------------------------------------------------------------------ 打印
def _pct(v, d=1):
    return "—" if v is None else f"{v*100:+.{d}f}%"


def report(st: dict) -> None:
    c = st["cover"]
    W = 78
    print("=" * W)
    print(f"  本地行情库统计    {c['first']} ~ {c['last']}")
    print("=" * W)
    print(f"  个股 {c['codes']:,} 只 / 已登记 {st['instruments']:,} 只   "
          f"K线 {c['bars']:,} 根   交易日 {st['trading_days']:,} 天")
    print(f"  全程在市 {st.get('full_history_codes',0):,} 只   "
          f"已退市或停更 {st.get('stale_codes',0):,} 只")

    print(f"\n【市场构成】  ST {st['st_count']} 只")
    print("  " + "   ".join(f"{k} {v}" for k, v in st["by_board"].items()))
    print("  近年上市：" + "  ".join(f"{y}年 {n}只"
                                for y, n in list(st["by_listed_year"].items())[:6]))

    print(f"\n【各年度全景】")
    print(f"  {'年份':<6}{'交易日':>6}{'个股数':>7}{'中位涨幅':>10}{'均值涨幅':>10}"
          f"{'上涨':>7}{'下跌':>7}{'翻倍':>6}{'腰斩':>6}")
    print("  " + "-" * (W - 4))
    for d in st["yearly_dist"]:
        yr = next((y for y in st["by_year"] if y[0] == d["year"]), None)
        print(f"  {d['year']:<6}{yr[1] if yr else 0:>6}{d['n']:>7}"
              f"{_pct(d['median']):>10}{_pct(d['mean']):>10}"
              f"{d['up']:>7}{d['down']:>7}{d['gt100']:>6}{d['lt50']:>6}")

    print(f"\n【指数分年度】")
    names = list(st["indices"])
    years = sorted({y for v in st["indices"].values() for y in v})
    print(f"  {'年份':<6}" + "".join(f"{n:>11}" for n in names))
    print("  " + "-" * (W - 4))
    for y in years:
        print(f"  {y:<6}" + "".join(f"{_pct(st['indices'][n].get(y)):>11}"
                                    for n in names))

    print(f"\n【涨跌停频率】  按 pct_chg 近似判定（20cm 板单独处理）")
    print(f"  {'年份':<6}{'涨停次数':>10}{'占比':>8}{'跌停次数':>10}{'占比':>8}")
    print("  " + "-" * (W - 4))
    for y, up, dn, tot in st["limit_stats"]:
        print(f"  {y:<6}{up:>10,}{up/tot:>8.2%}{dn:>10,}{dn/tot:>8.2%}")

    if "liquidity" in st:
        print(f"\n【近 90 日分位数】")
        L, V, P = st["liquidity"], st["volatility"], st["price"]
        print(f"  {'分位':<6}{'日均成交额':>14}{'年化波动率':>12}{'股价':>10}")
        print("  " + "-" * (W - 4))
        for p in (10, 25, 50, 75, 90):
            print(f"  p{p:<5}{L[f'p{p}']/1e8:>12.2f}亿{V[f'p{p}']:>11.0%}"
                  f"{P[f'p{p}']:>10.2f}")
        print(f"  股价 >400 元的股票 {st['price_over400']} 只"
              f"（20万5仓时一手就买不起）")

    if "suspend" in st:
        s = st["suspend"]
        print(f"\n【停牌】全程在市的股票中，缺失交易日占比："
              f"中位 {s['median']:.2%}  均值 {s['mean']:.2%}  "
              f"90分位 {s['p90']:.2%}")
        print(f"  缺失超过 10% 的有 {s['over10pct']} 只")

    # ---- 从数据里读出来的、会改变策略设计的几条 ----
    print(f"\n{'=' * W}\n  这些数字对策略意味着什么\n{'=' * W}")
    skew = [d for d in st["yearly_dist"] if d["mean"] > d["median"]]
    print(f"  · 收益右偏：{len(skew)}/{len(st['yearly_dist'])} 个年份里"
          f"「均值 > 中位数」，即少数大赢家拉高了整体。")
    worst = min(st["yearly_dist"], key=lambda d: d["median"])
    best = max(st["yearly_dist"], key=lambda d: d["median"])
    print(f"    最差 {worst['year']} 年中位 {_pct(worst['median'])}，"
          f"最好 {best['year']} 年中位 {_pct(best['median'])}——"
          f"年份差异远大于选股差异，所以择时比选股重要。")
    tot_up = sum(u for _, u, _, t in st["limit_stats"])
    tot_bar = sum(t for _, _, _, t in st["limit_stats"])
    print(f"  · 涨停占全部交易日的 {tot_up/tot_bar:.2%}——"
          f"追涨停策略的信号有相当比例根本买不进。")
    if "suspend" in st:
        print(f"  · 停牌中位 {st['suspend']['median']:.1%}——"
              f"回测若不建模停牌，会假设你随时能买卖。")
