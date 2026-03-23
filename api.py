"""
api.py — FastAPI 接口
提供 HTTP 接口，让 OpenClaw / 其他服务查询猫的位置

启动方式：
  uvicorn api:app --host 0.0.0.0 --port 8080

接口列表：
  GET /where_is_cat    → 猫现在在哪里
  GET /history         → 最近检测记录
  GET /zones           → 区域定义列表
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import time

from inference import infer_location, format_last_seen
from logger import get_recent, get_last_n
from zones import ZONES

app = FastAPI(
    title="🐱 CatWatch API",
    description="实时追踪猫的位置",
    version="0.1.0",
)


@app.get("/where_is_cat", summary="猫现在在哪里")
def where_is_cat():
    """
    推断猫当前所在区域
    
    返回示例：
    ```json
    {
      "location": "沙发",
      "confidence": "high",
      "confidence_score": 0.82,
      "last_seen": "12秒前",
      "last_seen_zone": "沙发",
      "sample_count": 8,
      "answer": "🐱 猫在沙发，12秒前看到的（置信度：高）"
    }
    ```
    """
    result = infer_location()

    # 格式化"最后见到"
    last_seen_str = format_last_seen(result["last_seen_ago"])

    # 置信度中文
    conf_zh = {"high": "高", "medium": "中", "low": "低", "none": "无"}.get(
        result["confidence"], result["confidence"]
    )

    # zone英文 → 中文显示映射
    zone_zh = {
        "balcony": "阳台", "bedroom": "卧室", "living_room": "客厅",
        "sofa": "沙发", "tv_cabinet": "电视柜", "kitchen": "厨房",
        "bathroom": "卫生间", "walk_in_closet": "衣柜",
        "utility": "杂物间", "hallway": "走廊", "unknown": "未知区域",
    }
    location_zh = zone_zh.get(result["location"], result["location"])

    # 自然语言回答
    if result["confidence"] == "none":
        answer = "🤷 不知道猫在哪里，最近没有检测到"
    else:
        answer = (
            f"🐱 猫在{location_zh}，"
            f"{last_seen_str}看到的"
            f"（置信度：{conf_zh}）"
        )

    return {
        **result,
        "last_seen": last_seen_str,
        "answer": answer,
        "timestamp": time.time(),
    }


@app.get("/history", summary="最近检测记录")
def history(
    seconds: int = Query(300, description="往前看多少秒（默认5分钟）"),
    limit: int = Query(50, description="最多返回多少条"),
):
    """返回最近的检测记录"""
    records = get_recent(seconds)[:limit]
    return {
        "count": len(records),
        "seconds": seconds,
        "records": records,
    }


@app.get("/zones", summary="区域定义")
def list_zones():
    """返回所有已定义的区域"""
    return {
        "count": len(ZONES),
        "zones": {name: {"polygon": pts} for name, pts in ZONES.items()},
    }


@app.get("/health", summary="健康检查")
def health():
    return {"status": "ok", "time": time.time()}
