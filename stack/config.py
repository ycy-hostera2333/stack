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
#   2. 一字板：开 = 高 = 低，且离未取整的涨跌停价不到 LIMIT_ADJ_TOL。
#      这一道是给**前复权的历史价**的：那是「原始价 × 复权因子」再取到分，按复权后的
#      前收重算涨停价又取一次整，两次舍入能差出一分半，第 1 道会漏掉约六分之一的
#      真涨停（送转多、复权因子小的票尤其明显）。
#
#      门槛必须是「开 = 高 = 低」，不能只看「开 = 高」：开 = 高 意味着全天没高过开盘，
#      收盘 <= 开盘——拿它当判据，挡掉的全是当天下跌的买入（卖出一侧同理，挡掉的全是
#      当天反弹的卖出），等于用当天的结果筛交易，回测反而偏乐观。开 = 高 = 低 时收盘
#      必然等于开盘，当天涨跌恒为零，这个判据不携带任何关于结果的信息。
#      代价：前复权历史上「开在涨停、盘中又打开」的那种涨停，仍按第 1 道判，约六分之一
#      判不出来——不知道复权因子就分不清，而用当天高低价去补又会引入上面的前视。
# 2026-10 在原始价、两位/三位小数前复权、未取整前复权上各 40 万例模拟：一字板漏判 0，
# 原始价上差一分、两分的正常开盘误挡 0。
LIMIT_MARGIN = (0.005, 0.005)      # (元, 占前收比例的上限)
LIMIT_ADJ_TOL = 0.02               # 元。一字板本身不带方向信息，不必按价格缩小


def limit_price(prev_close, limit, up: bool = True):
    """交易所口径的涨/跌停价：前收 ×(1±幅度)，四舍五入（half-up）到分。

    标量、numpy 数组、pandas 对象都能用。加一个极小量是为了抵消二进制浮点的误差：
    10.15×1.1 在浮点里可能是 11.164999…，按交易所口径它是 11.165，应当进位到 11.17。
    传进来的价格应当是 float64——float32 的误差会大过这个极小量。
    """
    raw = prev_close * (1 + limit) if up else prev_close * (1 - limit)
    return np.floor(raw * 100 + 0.5 + 1e-6) / 100


def _exact_margin(prev_close):
    return np.minimum(LIMIT_MARGIN[0], LIMIT_MARGIN[1] * prev_close)


def _one_price(px, high, low) -> bool:
    """开 = 高 = 低：全天一个价。"""
    return (high is not None and low is not None
            and px >= high - 1e-9 and px <= low + 1e-9)


def hit_limit_up(px: float, prev_close: float, limit: float,
                 high: float | None = None, low: float | None = None) -> bool:
    """以 px 开盘成交是否撞在涨停价上（视为买不进）。

    high/low 是当天最高/最低价，两个都给了才启用第 2 道判据（一字板）。
    """
    if px >= limit_price(prev_close, limit, True) - _exact_margin(prev_close):
        return True
    return bool(_one_price(px, high, low)
                and px >= prev_close * (1 + limit) - LIMIT_ADJ_TOL)


def hit_limit_down(px: float, prev_close: float, limit: float,
                   high: float | None = None, low: float | None = None) -> bool:
    """以 px 开盘成交是否撞在跌停价上（视为卖不出）。参数同 hit_limit_up。"""
    if px <= limit_price(prev_close, limit, False) + _exact_margin(prev_close):
        return True
    return bool(_one_price(px, high, low)
                and px <= prev_close * (1 - limit) + LIMIT_ADJ_TOL)


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
