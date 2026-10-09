"""网页上写的自定义策略：存库、编译、注册。

与内置策略的唯一区别是代码的**来源**：内置的写在 `stack/strategies/builtin.py`，
自定义的存 SQLite 的 `user_strategies` 表，服务启动时（或刚保存完）动态编译进
同一个 `REGISTRY`。编译进来之后两者完全等价——回测、每日信号、模拟盘、命令行
都不需要知道它从哪来。

保存前必须过三道关，因为这三类错误在本项目里全是**静默的**：不抛异常，
只是让回测结果变得好看，或者让策略干脆不出信号：

  1. 语法/运行错误。`entry()` 里写错列名的 KeyError 会被引擎的
     `except Exception: continue` 吞掉（见 `backtest/engine.py`），
     表现为「这个策略从来不出信号」，比报错难查得多。
  2. 前视偏差。`shift(-1)`、`cummax()`、对整段做 rank/归一化都会用到未来数据，
     回测收益凭空变好，而且自己几乎不可能发现。
  3. 返回值不合规。返回裸 numpy 数组时回测照跑（引擎按位置取值），
     但模拟盘用 `.loc[prev]` 定位，会静默跳过该股票——两边结果不一致，
     不报错。索引被 reset 过也一样。

所以 `check_code()` 会在合成的行情上真跑一遍 `prepare/entry/exit/score`，
再用「截断尾部重算、比对重叠区」的办法检出前视偏差，判据与自检里那条
指标检查完全一致。
"""
from __future__ import annotations

import inspect
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..config import ROOT
from ..data import store
from .base import (REGISTRY, Strategy, cross_above, cross_below, hysteresis,
                   index_above_ma, safe)

# 编译时用的文件名。报错的行号靠它从 traceback 里定位到用户代码那一层。
CODE_FILE = "<策略代码>"

# 标识：小写字母开头，只含小写字母/数字/下划线。它同时是注册名和库里主键。
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# 展示内置策略源码时拼在前面的头。用户「以此为模板」把这整段复制进编辑器，
# 改完保存即可——代码自带 import，脱离应用的命名空间也说得通。
BUILTIN_HEADER = '''"""从内置策略复制出来的模板。

df 里已经备好的指标列见 indicators.add_common：
  ma5/10/20/60/120、vol_ma20、vol_ratio、rsi14、atr14、atr_pct、
  high20、low20、dd、mom20/60/120、vol20
需要 MACD/布林带/MA250 时覆写 prepare() 调 ind.add_extended(super().prepare(df))；
别的周期的均线/动量用 ind.add_periods(df, ma=(30,), mom=(90,))。

⚠ 下面的 `indicators = (...)` 只让 df 里出现这个策略自己用到的那几列（为了快）。
改成用别的列时，把列名加进去，或者整行删掉恢复成全部都算。


entry/exit/score 只允许使用当日及之前的信息——用了未来数据，
回测会假得离谱，而且保存时会直接被拒。
"""

import numpy as np
import pandas as pd

from stack import indicators as ind
from stack.strategies.base import (Strategy, safe, cross_above, cross_below,
                                   hysteresis, index_above_ma)


'''


class CodeError(Exception):
    """用户代码的问题。line 为 None 表示定位不到具体行（比如参数校验失败）。"""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(message)
        self.message = message
        self.line = line


def _namespace(name: str, declared: list) -> dict[str, Any]:
    """用户代码能看到的名字。

    不是安全边界（本机自用、只监听 127.0.0.1，也允许自己 import），而是把
    「我该用什么」摆明白：基础类、几个信号组合工具、整套指标函数。

    `register` 是**局部**版本：内置策略源码里那行 `@register` 如果作用到全局
    注册表，复制过来改一份就会把同名的内置策略顶掉，而且没人会注意到。
    """
    def register_scoped(cls=None):
        if cls is None:                       # 也兼容 @register() 的写法
            def deco(c):
                declared.append(c)
                return c
            return deco
        declared.append(cls)
        return cls

    ns: dict[str, Any] = {
        "__name__": f"user_strategy_{name}",
        "__doc__": None,
        "np": np, "pd": pd, "ind": ind,
        "Strategy": Strategy, "register": register_scoped,
        "safe": safe, "cross_above": cross_above, "cross_below": cross_below,
        "hysteresis": hysteresis, "index_above_ma": index_above_ma,
    }
    for fn in ("sma", "ema", "rsi", "macd", "atr", "bollinger", "rolling_high",
               "rolling_low", "drawdown", "momentum", "volatility", "slope",
               "add_common", "add_extended"):
        ns[fn] = getattr(ind, fn)
    return ns


def _line_of(exc: BaseException) -> int | None:
    """异常落在用户代码里的行号。

    traceback 里文件名为 CODE_FILE 的那一层就是用户代码（最靠内的一层），
    其余层是 pandas / 引擎自己的。没有就返回 None，界面上就不显示行号。
    """
    line, tb = None, exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == CODE_FILE:
            line = tb.tb_lineno
        tb = tb.tb_next
    return line
def _pick_class(ns: dict, declared: list) -> type[Strategy]:
    """挑出代码里定义的策略类。

    只看**这份代码里定义**的（比对 `__module__`），否则命名空间里预置的
    Strategy 自己也会被算成候选。优先用 @register 记下来的那一个。
    """
    candidates = [c for c in declared if isinstance(c, type)]
    if not candidates:
        mod = ns.get("__name__")
        candidates = [v for v in ns.values()
                      if isinstance(v, type) and issubclass(v, Strategy)
                      and v is not Strategy and getattr(v, "__module__", "") == mod]
    if not candidates:
        raise CodeError("代码里没有找到策略类。需要定义一个继承 Strategy 的类，"
                        "例如 class MyStrategy(Strategy): ...")
    if len(candidates) > 1:
        names = "、".join(sorted(c.__name__ for c in candidates))
        raise CodeError(f"代码里定义了 {len(candidates)} 个策略类（{names}），"
                        "只能有一个。")
    return candidates[0]


def compile_code(code: str, name: str) -> type[Strategy]:
    """把源码编译成策略类。只编译，不注册、不落库。出错抛 CodeError。"""
    if not code or not code.strip():
        raise CodeError("代码是空的。可以在「内置策略」里点「以此为模板」抄一份改。")
    declared: list = []
    ns = _namespace(name, declared)
    try:
        exec(compile(code, CODE_FILE, "exec"), ns)
    except SyntaxError as e:
        raise CodeError(f"语法错误：{e.msg}", line=e.lineno) from None
    except Exception as e:
        # 类体/装饰器里就可能抛：defaults 写成 list、模块级调用了不存在的函数…
        raise CodeError(f"导入失败：{type(e).__name__}: {e}",
                        line=_line_of(e)) from None

    cls = _pick_class(ns, declared)
    # 这几个是引擎/界面依赖的契约，缺一个都会在后面变成静默错误
    if cls.entry is Strategy.entry:
        raise CodeError("必须实现 entry(df)：返回买入信号（与 df 等长的布尔 Series）")
    if cls.exit is Strategy.exit:
        raise CodeError("必须实现 exit(df)：返回卖出信号（与 df 等长的布尔 Series）。"
                        "不想设结构性离场就写 return safe(pd.Series(False, "
                        "index=df.index))")
    if not isinstance(getattr(cls, "defaults", None), dict):
        # 基类里的 defaults/param_meta 是 dataclass 的 default_factory，不会变成
        # 类属性，所以「不需要参数」的子类不写这两行时实例化会 AttributeError。
        # 这里补成空 dict，让它正常跑。
        if getattr(cls, "defaults", None) is None:
            cls.defaults = {}
        else:
            raise CodeError('defaults 必须是 dict，例如 defaults = {"n": 20}')
    if getattr(cls, "param_meta", None) is None:
        cls.param_meta = {}
    elif not isinstance(cls.param_meta, dict):
        raise CodeError("param_meta 必须是 dict（给界面用的参数说明）；没有就删掉这一行")
    return cls


def builtin_names() -> set[str]:
    """当前注册表里所有内置策略的名字。自定义策略不能占用这些名字。"""
    return {n for n, c in REGISTRY.items()
            if getattr(c, "origin", "builtin") == "builtin"}


def check_name(name: str) -> str:
    """校验标识（注册名）。返回规范化后的名字，不合法抛 CodeError。"""
    name = (name or "").strip()
    if not SLUG_RE.match(name):
        raise CodeError("标识只能用小写字母/数字/下划线、字母开头、2-32 个字符，"
                        "例如 my_ma_cross")
    if name in builtin_names():
        raise CodeError(f"{name!r} 是内置策略的名字，请换一个（例如 {name}_v2）——"
                        "同名会把内置策略顶掉，而回测/信号/模拟盘都还在用它")
    return name
# ------------------------------------------------------------------ 试跑与体检
# 合成数据 420 根日线：够算内置策略用到的绝大多数指标（最长是 ma120、
# 60 日通道）。固定随机种子，否则同一个策略有时报错有时不报，没人能复现。
SAMPLE_BARS = 420


def sample_df(n: int = SAMPLE_BARS) -> pd.DataFrame:
    """给试跑用的一段合成行情。价格是对数随机游走，列与
    store.load_daily() 返回的单只股票一致。

    索引刻意设成**日期字符串**（和 paper.py 里的 `d.index = idx` 一样）：
    模拟盘用 `.loc[prev]` 按日期取信号，索引对不上就会静默跳过整只股票。
    用默认的 RangeIndex 时，reset_index 过的返回值看起来和原来一模一样，
    这类错就检不出来了。
    """
    rng = np.random.default_rng(20260815)
    px = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    dates = pd.bdate_range("2019-01-02", periods=n)
    df = pd.DataFrame({
        "code": "600000", "date": dates,
        "open": px, "high": px * 1.012, "low": px * 0.988, "close": px,
        "volume": rng.integers(200_000, 900_000, n).astype(float),
        "amount": px * 300_000.0,
        "pct_chg": rng.normal(0, 2, n),
        "turnover": rng.uniform(0.5, 3.0, n),
    })
    df.index = dates.strftime("%Y-%m-%d")
    return df


class _DataMissing(Exception):
    """依赖的表/数据不存在。不是代码写错，只是本地库里还没有那份数据。"""


def _check_output(v, fn: str, index: pd.Index) -> pd.Series:
    """校验 entry/exit/score 的返回值。

    必须是带原索引、等长的 Series。理由不是洁癖：
      · 返回裸数组 —— 引擎按位置取值，照跑；模拟盘用 .loc[prev] 定位，
        会静默跳过这只股票，两边结果对不上而不报错。
      · reset_index 过的 Series —— 同上，模拟盘查不到日期，静默跳过。
    """
    if isinstance(v, pd.Series):
        if len(v) != len(index):
            raise CodeError(f"{fn}() 返回了 {len(v)} 个值，df 有 {len(index)} 行，"
                            "必须等长")
        if not v.index.equals(index):
            raise CodeError(f"{fn}() 返回的 Series 索引和 df 不一致，"
                            "不要 reset_index 或另建 Series——模拟盘靠 df 的日期"
                            "做 .loc 定位，索引对不上会静默跳过整只股票")
        return v
    raise CodeError(f"{fn}() 必须返回与 df 等长的 pandas Series，"
                    f"现在拿到的是 {type(v).__name__}。"
                    "布尔信号请用 safe(...) 包一下，打分直接返回 df 的某一列")


def _run_signals(strat: Strategy, df: pd.DataFrame) -> dict[str, pd.Series]:
    """按引擎的调用方式跑一遍 prepare/entry/exit/score。"""
    try:
        d = strat.prepare(df)
    except sqlite3.Error as e:
        raise _DataMissing(str(e)) from None
    except Exception as e:
        raise CodeError(f"prepare() 出错：{type(e).__name__}: {e}",
                        line=_line_of(e)) from None
    if not isinstance(d, pd.DataFrame):
        raise CodeError(f"prepare() 必须返回 DataFrame，现在返回的是 "
                        f"{type(d).__name__}")
    if len(d) != len(df) or not d.index.equals(df.index):
        raise CodeError("prepare() 不能改变行数或索引（只加列）。"
                        "要新列就直接 df[col] = ...，不要 reset_index 或过滤行")

    out: dict[str, pd.Series] = {}
    for fn in ("entry", "exit", "score"):
        try:
            v = getattr(strat, fn)(d)
        except sqlite3.Error as e:
            raise _DataMissing(str(e)) from None
        except Exception as e:
            # 这一句就是关键：引擎会把这个异常吞掉，只表现为「没有信号」
            raise CodeError(f"{fn}() 出错：{type(e).__name__}: {e}",
                            line=_line_of(e)) from None
        out[fn] = _check_output(v, fn, d.index)
    return out


def find_lookahead(strat: Strategy, cut: int = 5) -> str | None:
    """检出前视偏差（用到了未来数据）。有则返回说明，没有返回 None。

    判据与自检里那条指标检查一样：把最后 cut 根 K 线截掉重算，重叠区的信号
    必须一字不差。用了 shift(-1)、cummax()、`.max()`（少了 rolling）、对整段
    做 rank/归一化，都会在这里现形——而在回测里它们只会让收益变好看。
    """
    df = sample_df()
    full = _run_signals(strat, df)
    part = _run_signals(strat, df.iloc[:-cut])

    for fn in ("entry", "exit"):
        a = full[fn].fillna(False).astype(bool).to_numpy()[:-cut]
        b = part[fn].fillna(False).astype(bool).to_numpy()
        if not np.array_equal(a, b):
            i = int(np.flatnonzero(a != b)[0]) + 1
            return (f"{fn}() 用到了未来数据：截掉最后 {cut} 根 K 线重算后，"
                    f"第 {i} 根的结果变了。\n"
                    "常见写法：shift(-1)、cummax()/cumsum() 直接作用于全序列、"
                    "df['high'].max()（少了 rolling）、对整段做 rank/归一化。\n"
                    "这类写法在回测里只会让收益凭空变好，看不出错。")
    a = full["score"].fillna(-9.9).to_numpy()[:-cut]
    b = part["score"].fillna(-9.9).to_numpy()
    if not np.allclose(a, b):
        i = int(np.flatnonzero(~np.isclose(a, b))[0]) + 1
        return f"score() 用到了未来数据：截掉最后 {cut} 根 K 线后，第 {i} 根的打分变了。"
    return None
# ------------------------------------------------------------------ 体检与保存
def _blank_report() -> dict:
    return {"ok": False, "error": None, "line": None, "notes": [], "checks": [],
            "name": "", "label": "", "description": "", "defaults": {},
            "params_used": {}, "param_meta": {}, "warmup_bars": None,
            "declared_name": "", "code_lines": 0, "saved": False, "exists": False}


def analyze(code: str, name: str, strict_name: bool = True,
            params: dict | None = None) -> tuple[type[Strategy] | None, dict]:
    """编译 → 实例化 → 试跑 → 前视偏差检查。返回 (策略类, 报告)。

    不抛异常：调用方是界面，行号和原因要原样显示出来。

    strict_name=False 用于「试跑还没保存的代码」：那时 name 只是个标题，
    允许和内置策略重名（用户往往正是复制内置策略来改的）。

    params 是**当前要用的那组参数**（编辑器里填的），不是默认值。只按默认值
    体检会漏掉一整类错：参数被拼进列名时（如 df[f"ma{n}"]），n=20 恰好存在、
    n=30 不存在，默认值试跑通过、真正跑起来却让所有股票被静默跳过。
    """
    rep = _blank_report()
    rep["code_lines"] = len(code.splitlines())

    if strict_name:
        try:
            slug = check_name(name)
        except CodeError as e:
            rep["error"], rep["line"] = e.message, e.line
            return None, rep
    else:
        slug = (name or "custom").strip()
        if not SLUG_RE.match(slug):
            slug = "custom"
    rep["name"] = slug

    try:
        cls = compile_code(code, slug)
        rep["checks"].append("语法与类定义通过")
    except CodeError as e:
        rep["error"], rep["line"] = e.message, e.line
        return None, rep

    try:
        strat = cls(**(params or {}))
    except Exception as e:
        rep["error"] = (f"实例化失败（参数 {params or '默认值'}）："
                        f"{type(e).__name__}: {e}\n"
                        "（validate() 或未知参数名都会在这里报出来）")
        rep["line"] = _line_of(e)
        return None, rep

    rep["defaults"] = dict(getattr(cls, "defaults", None) or {})
    rep["params_used"] = dict(getattr(strat, "params", None) or {})
    rep["param_meta"] = dict(getattr(cls, "param_meta", None) or {})
    rep["label"] = str(getattr(cls, "label", "") or slug)
    rep["description"] = str(getattr(cls, "description", "") or "")
    rep["declared_name"] = str(getattr(cls, "name", "") or "")
    rep["checks"].append(f"参数校验通过（{len(rep['params_used'])} 个参数）")
    if rep["declared_name"] and rep["declared_name"] != slug:
        rep["notes"].append(
            f"代码里 name = {rep['declared_name']!r} 会被标识 {slug!r} 覆盖。"
            "这是有意的：复制内置策略改一份时，同名会把内置策略顶掉。")

    try:
        warm = int(strat.warmup_bars())
    except Exception as e:
        rep["error"] = f"warmup_bars() 出错：{type(e).__name__}: {e}"
        rep["line"] = _line_of(e)
        return None, rep
    if warm <= 0:
        rep["error"] = "warmup_bars() 必须返回正数（算出有效信号最少要几根 K 线）"
        return None, rep
    rep["warmup_bars"] = warm
    rep["checks"].append(f"预热窗口 {warm} 根 K 线")
    if warm > 400:
        rep["notes"].append(f"warmup_bars()={warm}：回测与每次信号扫描都要为此多加载"
                            f"{warm} 根历史，会明显变慢。只在真的需要长周期时才设这么大。")

    # 大盘择时：引擎会 reindex 到交易日并按布尔序列用，格式错了回测里直接抛异常
    if cls.market_regime is not Strategy.market_regime:
        dates = sample_df()["date"].dt.strftime("%Y-%m-%d").tolist()
        try:
            reg = strat.market_regime(dates)
        except Exception as e:
            rep["error"] = f"market_regime() 出错：{type(e).__name__}: {e}"
            rep["line"] = _line_of(e)
            return None, rep
        if reg is not None:
            if not isinstance(reg, pd.Series) or len(reg) != len(dates):
                rep["error"] = ("market_regime() 必须返回与 dates 等长的布尔 Series，"
                                "或 None（不做择时）")
                return None, rep
            rep["checks"].append("大盘择时返回格式正确")

    # 合成行情试跑：真跑一遍，让「写错列名」这类错误当场现形，而不是被引擎吞掉
    try:
        _run_signals(strat, sample_df())
        rep["checks"].append("合成行情试跑通过（prepare/entry/exit/score）")
        smoke_ok = True
    except _DataMissing as e:
        rep["notes"].append(f"跳过了合成行情试跑：依赖本地库里还没有的数据（{e}）。"
                            "这类策略请自己确认字段名写对了。")
        smoke_ok = False
    except CodeError as e:
        rep["error"], rep["line"] = e.message, e.line
        return None, rep

    if smoke_ok:
        bad = find_lookahead(strat)
        if bad:
            rep["error"] = bad
            return None, rep
        rep["checks"].append("无前视偏差（截掉尾部 5 根 K 线重算，重叠区一致）")
    else:
        rep["notes"].append("未能做前视偏差检查（试跑被跳过）")

    rep["ok"] = True
    return cls, rep


def validate(code: str, name: str, params: dict | None = None) -> dict:
    """只要报告，不保存。界面上的「检查」按钮。"""
    return analyze(code, name, True, params)[1]
def register_class(cls: type[Strategy], name: str,
                   label: str | None = None) -> None:
    """把编译好的类挂进全局注册表。名字以标识为准（见 analyze 里的说明）。"""
    cls.name = name
    cls.origin = "user"
    if label:
        cls.label = label
    if not getattr(cls, "label", ""):
        cls.label = name
    REGISTRY[name] = cls


def save(name: str, label: str, description: str, code: str,
         overwrite: bool = False, params: dict | None = None) -> dict:
    """校验 → 落库 → 注册。返回报告；ok=False 表示没保存。

    params 是编辑器里当前填的那组参数，只用于体检；落库的永远是代码本身
    （参数是运行时传的，不入库）。
    """
    cls, rep = analyze(code, name, True, params)
    if not rep["ok"] or cls is None:
        return rep
    slug = rep["name"]
    existing = _load_row(slug)
    if existing and not overwrite:
        rep["ok"] = False
        rep["exists"] = True
        rep["error"] = (f"已经有一个叫 {slug} 的自定义策略了"
                        f"（建于 {existing.get('created_at')}，"
                        f"最后改于 {existing.get('updated_at')}）。"
                        "要覆盖它就勾上「覆盖已有」。")
        return rep
    label = (label or "").strip() or rep["label"]
    description = (description or "").strip() or rep["description"]
    register_class(cls, slug, label)
    store.upsert_user_strategy(slug, label, description, code)
    load_errors.pop(slug, None)
    rep.update(saved=True, label=label, description=description)
    return rep


def remove(name: str, force: bool = False) -> dict:
    """删除自定义策略。

    被模拟盘账户用着的默认拦住：账户只存策略名，删掉之后那个账户下次推进时
    get_strategy 会抛 KeyError，这一期及之后的前向记录就断了——而前向记录断了
    补不回来（那是唯一不能事后调参的验证）。
    """
    if getattr(REGISTRY.get(name), "origin", None) != "user" and not _load_row(name):
        return {"ok": False, "error": f"{name!r} 不是自定义策略，删不了"}
    used = _accounts_using(name)
    if used and not force:
        return {"ok": False, "accounts": used,
                "error": f"模拟盘账户 {'、'.join(used)} 正在用它：删除后这些账户"
                         "推进会直接失败，而前向记录断了没法补。"
                         "确认要删就先把那些账户删掉。"}
    store.delete_user_strategy(name)
    REGISTRY.pop(name, None)
    load_errors.pop(name, None)
    return {"ok": True}


def _accounts_using(name: str) -> list[str]:
    try:
        from .. import paper
        return [a["account"] for a in paper.list_accounts()
                if a.get("strategy") == name]
    except Exception:
        # 这里只是护栏，不该因为账户表本身有问题就让人删不掉策略
        return []


def _load_row(name: str) -> dict | None:
    try:
        return store.load_user_strategy(name)
    except sqlite3.OperationalError:      # 表还没建起来
        return None
# ------------------------------------------------------------------ 加载与展示
_loaded = False
# 名字 → 加载失败原因。界面上必须能看到：否则表现只是「策略列表里少了一个」。
load_errors: dict[str, str] = {}


def ensure_loaded(force: bool = False) -> None:
    """把库里保存的策略编译进 REGISTRY。首次调用时执行，幂等。

    放在函数里而不是模块导入时：导入发生得比建库早（cli.py 的 import 在
    init_db() 之前），那时表还不存在。
    """
    global _loaded
    if _loaded and not force:
        return
    _loaded = True
    for row in _saved_rows(with_code=True):
        name = row["name"]
        try:
            cls = compile_code(row.get("code") or "", name)
            register_class(cls, name, row.get("label"))
            if row.get("description"):
                cls.description = row["description"]
            load_errors.pop(name, None)
        except CodeError as e:
            load_errors[name] = e.message
        except Exception as e:
            load_errors[name] = f"{type(e).__name__}: {e}"


def _saved_rows(with_code: bool = False) -> list[dict]:
    try:
        df = store.list_user_strategies(with_code=with_code)
    except sqlite3.OperationalError:
        return []                     # 表还没建 = 没有自定义策略，不是错误
    return df.to_dict("records") if not df.empty else []


def saved_overview() -> list[dict]:
    """给界面列表用的概览：不含代码，但带行数与加载失败原因。"""
    out = []
    for r in _saved_rows(with_code=True):
        code = r.get("code") or ""
        out.append({"name": r["name"], "label": r.get("label") or r["name"],
                    "description": r.get("description") or "",
                    "created_at": r.get("created_at"),
                    "updated_at": r.get("updated_at"),
                    "lines": len(code.splitlines()),
                    "load_error": load_errors.get(r["name"])})
    return out


def source_of(name: str) -> dict | None:
    """策略源码，给界面展示。

    内置策略是实时读真实 .py 文件（inspect），不做快照副本：副本迟早和真身
    对不上，而这个功能的用法恰恰是「照着展示的代码改一份」。
    """
    cls = REGISTRY.get(name)
    if getattr(cls, "origin", None) == "user" or cls is None:
        row = _load_row(name)
        if not row:
            return None
        err = load_errors.get(name)
        return {"name": name, "origin": "user",
                "label": row.get("label") or name,
                "description": row.get("description") or "",
                "code": row.get("code") or "",
                "updated_at": row.get("updated_at"), "load_error": err,
                "note": ("这份代码当前加载失败，修好再保存。" if err
                         else "这份代码存在本地库里，是当前真正在跑的那一份。")}

    try:
        body = inspect.getsource(cls)
        path = inspect.getsourcefile(cls)
        line = inspect.getsourcelines(cls)[1]
    except (OSError, TypeError):
        return None
    try:
        rel = str(Path(path).relative_to(ROOT)) if path else ""
    except ValueError:
        rel = path or ""
    return {"name": name, "origin": "builtin", "file": rel, "line": line,
            "label": getattr(cls, "label", name),
            "description": getattr(cls, "description", ""),
            "code": BUILTIN_HEADER + body, "load_error": None,
            "note": f"内置策略的源码，实时读自 {rel or '源码文件'}。"
                    "想改成自己的：复制到编辑器里改，换个标识保存即可"}


def strategy_from_code(code: str, name: str = "custom", **params) -> Strategy:
    """临时跑一份还没保存的代码（编辑器里的「试跑回测」）。

    不进注册表：试跑不该改名，也不该在策略下拉里留下东西。name 只当标题用。
    """
    slug = (name or "custom").strip()
    if not SLUG_RE.match(slug):
        slug = "custom"
    cls = compile_code(code, slug)
    cls.name = slug
    cls.origin = "ad_hoc"
    try:
        return cls(**params)
    except Exception as e:
        raise CodeError(f"参数不合法：{e}") from None