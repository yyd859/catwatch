"""
inference.py — 位置推断模块
根据历史检测记录，推断猫"现在在哪里"

核心逻辑：时间衰减加权投票
  - 越近的记录权重越高
  - 连续出现在同一区域，置信度更高
"""

import time
import math
from logger import get_recent, get_last_n


# ── 推断参数（可调） ──────────────────────────────
DECAY_HALF_LIFE = 60.0   # 衰减半衰期（秒）：60秒前的记录权重减半
LOOKBACK_SECS   = 600    # 最多往前看多少秒（10分钟）
MIN_WEIGHT      = 0.05   # 低于此权重的记录忽略


def time_weight(ts: float, now: float | None = None) -> float:
    """
    计算时间衰减权重
    公式：weight = 2^(-(now - ts) / half_life)
    
    例：
      0秒前  → weight=1.0
      60秒前 → weight=0.5
      120秒前 → weight=0.25
    """
    if now is None:
        now = time.time()
    elapsed = now - ts
    return math.pow(2.0, -elapsed / DECAY_HALF_LIFE)


def infer_location(logs: list[dict] | None = None) -> dict:
    """
    推断猫当前位置
    
    参数：
      logs — 检测记录列表（None 则自动从数据库读取最近记录）
    
    返回：
      {
        "location": "沙发",          # 推断位置
        "confidence": "high",        # 置信度等级：high/medium/low
        "confidence_score": 0.82,    # 数值置信度
        "last_seen_ago": 12,         # 最近一次检测距现在多少秒
        "last_seen_zone": "沙发",    # 最近一次检测的区域
        "sample_count": 8,           # 参与推断的记录条数
      }
    """
    now = time.time()

    # 读取数据
    if logs is None:
        logs = get_recent(LOOKBACK_SECS)

    if not logs:
        return {
            "location": "unknown",
            "confidence": "none",
            "confidence_score": 0.0,
            "last_seen_ago": None,
            "last_seen_zone": None,
            "sample_count": 0,
        }

    # 按时间加权投票
    zone_weights: dict[str, float] = {}
    total_weight = 0.0

    for record in logs:
        w = time_weight(record["timestamp"], now)
        if w < MIN_WEIGHT:
            continue
        # 乘上检测置信度
        w *= record["confidence"]
        zone = record["zone"]
        zone_weights[zone] = zone_weights.get(zone, 0.0) + w
        total_weight += w

    if not zone_weights or total_weight == 0:
        return {
            "location": "unknown",
            "confidence": "none",
            "confidence_score": 0.0,
            "last_seen_ago": None,
            "last_seen_zone": None,
            "sample_count": 0,
        }

    # 找得分最高的区域
    best_zone = max(zone_weights, key=zone_weights.get)
    best_score = zone_weights[best_zone] / total_weight  # 归一化到 0~1

    # 置信度等级
    if best_score >= 0.7:
        confidence_level = "high"
    elif best_score >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # 最近一次检测
    latest = max(logs, key=lambda r: r["timestamp"])
    last_seen_ago = int(now - latest["timestamp"])

    return {
        "location": best_zone,
        "confidence": confidence_level,
        "confidence_score": round(best_score, 3),
        "last_seen_ago": last_seen_ago,
        "last_seen_zone": latest["zone"],
        "sample_count": len(logs),
    }


def format_last_seen(seconds: int | None) -> str:
    """把秒数格式化成人话"""
    if seconds is None:
        return "从未检测到"
    if seconds < 60:
        return f"{seconds}秒前"
    if seconds < 3600:
        return f"{seconds // 60}分钟前"
    return f"{seconds // 3600}小时前"
