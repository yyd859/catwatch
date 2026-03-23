"""
logger.py — 检测日志模块
把每次猫的检测结果写入 SQLite 数据库
"""

import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass

DB_PATH = Path(__file__).parent / "data" / "catwatch.db"


@dataclass
class Detection:
    """单次检测记录"""
    timestamp: float      # Unix 时间戳
    track_id: int         # YOLO 追踪 ID（同一只猫尽量保持稳定）
    zone: str             # 所在区域
    confidence: float     # 置信度 0~1
    cx: float             # 中心点 x
    cy: float             # 中心点 y


def init_db():
    """初始化数据库，首次运行时建表"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   REAL NOT NULL,
            track_id    INTEGER,
            zone        TEXT NOT NULL,
            confidence  REAL NOT NULL,
            cx          REAL,
            cy          REAL
        )
    """)
    # 加索引，查询快
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON detections(timestamp)")
    conn.commit()
    conn.close()
    print(f"✅ 数据库已初始化：{DB_PATH}")


def log_detection(det: Detection):
    """写入一条检测记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO detections (timestamp, track_id, zone, confidence, cx, cy) VALUES (?,?,?,?,?,?)",
        (det.timestamp, det.track_id, det.zone, det.confidence, det.cx, det.cy)
    )
    conn.commit()
    conn.close()


def get_recent(seconds: int = 300) -> list[dict]:
    """
    获取最近 N 秒的检测记录
    
    参数：
      seconds — 往前看多少秒（默认5分钟）
    
    返回：
      检测记录列表，每条是个 dict
    """
    since = time.time() - seconds
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections WHERE timestamp > ? ORDER BY timestamp DESC",
        (since,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_last_n(n: int = 20) -> list[dict]:
    """获取最新 N 条记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY timestamp DESC LIMIT ?",
        (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
