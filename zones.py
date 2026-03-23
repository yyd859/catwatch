"""
zones.py — 区域映射模块
把像素坐标 (cx, cy) 映射到人类可读的区域名称

使用方法：
  1. 打开你家的地图图片
  2. 用图片查看器找到各区域的角点坐标
  3. 填入下面的 ZONES 字典（多边形顶点，顺序随意）
"""

import numpy as np

# ────────────────────────────────────────────────
# 📐 区域定义（根据扫地机器人地图 map/map.jpg 标定）
# 图片尺寸约 510 x 720 像素，左上角为原点 (0,0)
# 格式：区域名 → 多边形顶点列表 [[x1,y1],[x2,y2],...]
# ────────────────────────────────────────────────
ZONES: dict[str, list[list[int]]] = {
    # 阳台（左上角，卧室北侧）
    "balcony": [
        [30, 30], [230, 30], [230, 140], [30, 140]
    ],
    # 卧室（左侧大面积区域）
    "bedroom": [
        [30, 140], [250, 140], [250, 380], [30, 380]
    ],
    # 客厅/餐厅（右侧纵向区域，含沙发和电视柜）
    "living_room": [
        [270, 30], [460, 30], [460, 370], [270, 370]
    ],
    # 沙发（客厅右侧靠墙）
    "sofa": [
        [380, 150], [450, 150], [450, 300], [380, 300]
    ],
    # 电视柜区域（客厅左侧，靠近卧室墙）
    "tv_cabinet": [
        [255, 100], [310, 100], [310, 280], [255, 280]
    ],
    # 厨房（右下方）
    "kitchen": [
        [270, 370], [480, 370], [480, 600], [270, 600]
    ],
    # 卫生间（左下方）
    "bathroom": [
        [30, 430], [200, 430], [200, 620], [30, 620]
    ],
    # 步入式衣柜（卧室下方）
    "walk_in_closet": [
        [30, 380], [140, 380], [140, 440], [30, 440]
    ],
    # 杂物间/洗衣烘干区
    "utility": [
        [230, 300], [310, 300], [310, 410], [230, 410]
    ],
    # 中央走廊/过渡区（连接各房间）
    "hallway": [
        [140, 380], [270, 380], [270, 670], [140, 670]
    ],
}

# 找不到时的默认区域
UNKNOWN_ZONE = "unknown"


def point_in_polygon(px: float, py: float, polygon: list[list[int]]) -> bool:
    """射线法判断点是否在多边形内"""
    n = len(polygon)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def map_to_zone(cx: float, cy: float) -> str:
    """
    输入猫的中心点坐标，返回所在区域名称
    
    参数：
      cx, cy — 检测框中心点的像素坐标
    
    返回：
      区域名称字符串，如 "沙发"、"猫树" 等
    """
    for zone_name, polygon in ZONES.items():
        if point_in_polygon(cx, cy, polygon):
            return zone_name
    return UNKNOWN_ZONE


def get_zone_center(zone_name: str) -> tuple[float, float] | None:
    """返回指定区域的几何中心（调试用）"""
    if zone_name not in ZONES:
        return None
    pts = np.array(ZONES[zone_name])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())
