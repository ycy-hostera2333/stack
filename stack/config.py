"""全局配置与 A 股交易规则常量。"""
from __future__ import annotations

from pathlib import Path

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


# 涨跌停判定的容差。交易所的涨跌停价是「前收 ×(1±幅度)」四舍五入到分：
# 前收 10.13 的涨停价是 11.14，不是 11.143。早先用 1e-6 的容差去比，
# 凡是向下取整的涨停（约一半）都判不出来，回测照样在一字板上「买到」，
# 结果系统性偏乐观；行情源给的前复权价本身也只到分，两头各有半分的舍入。
# 所以容差取一分钱；低价股不超过前收的 0.5%，免得把 +9% 的开盘也当成涨停。
LIMIT_TICK = 0.01


def _limit_tol(prev_close: float) -> float:
    return min(LIMIT_TICK, 0.005 * prev_close)


def hit_limit_up(px: float, prev_close: float, limit: float) -> bool:
    """以 px 成交是否撞在涨停价上（开盘一字涨停视为买不进）。"""
    return px >= prev_close * (1 + limit) - _limit_tol(prev_close)


def hit_limit_down(px: float, prev_close: float, limit: float) -> bool:
    """以 px 成交是否撞在跌停价上（开盘一字跌停视为卖不出）。"""
    return px <= prev_close * (1 - limit) + _limit_tol(prev_close)


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
