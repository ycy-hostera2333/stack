"""全局配置与 A 股交易规则常量。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "market.db"
WEB_DIR = ROOT / "web"

DATA_DIR.mkdir(exist_ok=True)

# 本地行情库起始日期。往前拉越多回测越可信，但首次全量同步越慢。
HISTORY_START = "20180101"

# ---------------------------------------------------------------- 交易成本
COMMISSION_RATE = 0.00025      # 佣金 万2.5，双边
COMMISSION_MIN = 5.0           # 单笔最低 5 元
STAMP_TAX_RATE = 0.0005        # 印花税 千0.5，仅卖出
TRANSFER_FEE_RATE = 0.00001    # 过户费 十万分之1，双边
LOT_SIZE = 100                 # 一手 100 股

# ---------------------------------------------------------------- 涨跌停幅度
# 按代码前缀判定所属板块，ST 股另行折半。
BOARD_RULES = {
    "沪主板": {"prefixes": ("600", "601", "603", "605"), "limit": 0.10},
    "科创板": {"prefixes": ("688", "689"), "limit": 0.20},
    "深主板": {"prefixes": ("000", "001", "002", "003"), "limit": 0.10},
    "创业板": {"prefixes": ("300", "301"), "limit": 0.20},
    "北交所": {"prefixes": ("43", "83", "87", "88", "92", "920"), "limit": 0.30},
}


def classify_board(code: str) -> str:
    """由 6 位代码判断板块。"""
    code = str(code).zfill(6)
    for board, rule in BOARD_RULES.items():
        if code.startswith(rule["prefixes"]):
            return board
    return "未知"


def price_limit(code: str, name: str = "") -> float:
    """返回该股当日涨跌停幅度（小数）。ST 股主板减半为 5%。"""
    board = classify_board(code)
    limit = BOARD_RULES.get(board, {}).get("limit", 0.10)
    if "ST" in str(name).upper().replace(" ", ""):
        # 科创板/创业板 ST 仍为 20%，主板 ST 为 5%
        return limit if limit > 0.10 else 0.05
    return limit


# 涨跌停判定。交易所的涨跌停价是「前收 ×(1±幅度)」**四舍五入到分**：
# 前收 10.13 的涨停价是 11.14，不是 11.143。早先拿未取整的 前收×1.1 去比、容差 1e-6，
# 凡是向下取整的涨停（约一半）都判不出来，回测照样在一字板上「买到」，系统性偏乐观。
#
# 两道判据，任一成立即视为撞板：
#   1. 精确价：按交易所规则算出涨跌停价，留半分钱余量（只为吸收浮点误差）。
#      原始价上真正的涨跌停一个不漏，差一分的正常开盘一个不挡。
#   2. 开盘即最高（跌停：开盘即最低）且离未取整的涨停价不到 LIMIT_ADJ_TOL。
#      这一道是给**前复权的历史价**的：那是「原始价 × 复权因子」再取到分，
#      按复权后的前收再算一遍涨停价、再取一次整，两次舍入能差出近两分钱，
#      第 1 道会漏掉约六分之一的真涨停（送转股多的票尤其明显）。真涨停时当天不可能
#      高过开盘价，这个独立信号让放宽的容差不至于误伤正常开盘。
#      代价：原始价上「开盘差一分到涨停、且全天没高过开盘」也会被当成涨停——
#      极少见，而且是保守方向（少买），不会让回测偏乐观。
# 2026-10 在原始价、两位/三位小数前复权、未取整前复权、现金分红前复权五种模型上
# 各 40 万例模拟：真涨停漏判 ≤ 0.01%，原始价上差一分且盘中更高的开盘误挡 0%。
# (元, 占前收比例的上限)：低价股按比例缩小，免得把 +9% 的开盘也当成涨停
LIMIT_MARGIN = (0.005, 0.005)
LIMIT_ADJ_TOL = (0.016, 0.008)


def limit_price(prev_close, limit, up: bool = True):
    """交易所口径的涨/跌停价：前收 ×(1±幅度)，四舍五入（half-up）到分。

    标量、numpy 数组、pandas 对象都能用。加一个极小量是为了抵消二进制浮点的误差：
    10.15×1.1 在浮点里可能是 11.164999…，按交易所口径它是 11.165，应当进位到 11.17。
    传进来的价格应当是 float64——float32 的误差会大过这个极小量。
    """
    raw = prev_close * (1 + limit) if up else prev_close * (1 - limit)
    return np.floor(raw * 100 + 0.5 + 1e-6) / 100


def _margin(prev_close, tol):
    yuan, pct = tol
    return np.minimum(yuan, pct * prev_close)


def hit_limit_up(px: float, prev_close: float, limit: float,
                 high: float | None = None) -> bool:
    """以 px 开盘成交是否撞在涨停价上（视为买不进）。high 是当天最高价，给了才启用第 2 道判据。"""
    if px >= limit_price(prev_close, limit, True) - _margin(prev_close, LIMIT_MARGIN):
        return True
    return bool(high is not None and px >= high - 1e-9
                and px >= prev_close * (1 + limit) - _margin(prev_close, LIMIT_ADJ_TOL))


def hit_limit_down(px: float, prev_close: float, limit: float,
                   low: float | None = None) -> bool:
    """以 px 开盘成交是否撞在跌停价上（视为卖不出）。low 是当天最低价，给了才启用第 2 道判据。"""
    if px <= limit_price(prev_close, limit, False) + _margin(prev_close, LIMIT_MARGIN):
        return True
    return bool(low is not None and px <= low + 1e-9
                and px <= prev_close * (1 - limit) + _margin(prev_close, LIMIT_ADJ_TOL))


def buy_cost(amount: float) -> float:
    """买入总费用（佣金 + 过户费）。"""
    return max(amount * COMMISSION_RATE, COMMISSION_MIN) + amount * TRANSFER_FEE_RATE


def sell_cost(amount: float) -> float:
    """卖出总费用（佣金 + 印花税 + 过户费）。"""
    return (
        max(amount * COMMISSION_RATE, COMMISSION_MIN)
        + amount * STAMP_TAX_RATE
        + amount * TRANSFER_FEE_RATE
    )
