"""
detection.py — 猫检测主循环
使用 YOLOv8 追踪视频流中的猫，输出检测记录

用法：
  python detection.py                     # 使用摄像头（树莓派）
  python detection.py --source cat.mp4    # 使用视频文件（测试用）
  python detection.py --source cat.jpg    # 使用图片（快速验证）
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

from zones import map_to_zone
from logger import init_db, log_detection, Detection

# YOLO 中"cat"对应的类别 ID（COCO 数据集第 15 类）
CAT_CLASS_ID = 15
DOG_CLASS_ID = 16  # 黑白花纹猫有时被误识别为狗，一并追踪
PET_CLASS_IDS = [CAT_CLASS_ID, DOG_CLASS_ID]

# 模型选择：
#   yolov8n.pt  — 最小最快，漏检多（不推荐）
#   yolov8s.pt  — 小模型，平衡速度和精度（树莓派可用）
#   yolov8m.pt  — 中等，精度明显提升（推荐）
#   yolov8l.pt  — 大模型，精度最高，但慢
MODEL_PATH = "yolov8m.pt"


def run(source=0, show=True, conf_threshold=0.15, debug=False):
    """
    主检测循环
    
    参数：
      source         — 视频源（0=摄像头，或文件路径）
      show           — 是否显示实时画面
      conf_threshold — 最低置信度（低于此值忽略）
    """
    # 初始化
    init_db()
    model = YOLO(MODEL_PATH)
    print(f"✅ 模型加载完成：{MODEL_PATH}")
    print(f"📹 视频源：{source}")
    print("🐱 开始检测猫...\n")

    # debug 模式：不过滤类别，看猫被误识别成什么
    # 正常模式：同时追踪 cat + dog（黑白花猫易被误识别为狗）
    track_classes = None if debug else PET_CLASS_IDS

    # 用 track 模式（保持 track_id 稳定）
    for result in model.track(
        source=source,
        stream=True,
        classes=track_classes,
        conf=conf_threshold,
        show=show,
        verbose=False,
    ):
        ts = time.time()
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            # 置信度
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            # 追踪 ID（可能为 None，用 -1 兜底）
            track_id = int(box.id[0]) if box.id is not None else -1

            # 中心点坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # debug 模式：打印所有检测到的类别
            if debug:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_id != CAT_CLASS_ID:
                    print(f"  [DEBUG] 非猫检测: cls={cls_name}({cls_id}) conf={conf:.2f} 位置=({cx:.0f},{cy:.0f})")
                    continue  # debug 模式下跳过非猫

            # 区域映射
            zone = map_to_zone(cx, cy)

            # 写入数据库
            det = Detection(
                timestamp=ts,
                track_id=track_id,
                zone=zone,
                confidence=conf,
                cx=cx,
                cy=cy,
            )
            log_detection(det)

            # 控制台输出
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"🐱 ID={track_id:3d} | "
                f"区域={zone:8s} | "
                f"置信度={conf:.2f} | "
                f"位置=({cx:.0f}, {cy:.0f})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猫位置检测")
    parser.add_argument("--source", default=0, help="视频源（0=摄像头，或视频/图片路径）")
    parser.add_argument("--no-show", action="store_true", help="不显示实时画面（无头模式，适合树莓派）")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度（默认0.25，降低可减少漏检）")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="YOLO模型路径（默认yolov8s.pt）")
    parser.add_argument("--debug", action="store_true", help="调试模式：显示所有检测类别，找出猫被误识别成啥")
    args = parser.parse_args()

    if args.model != MODEL_PATH:
        import ultralytics
        # Allow overriding model at runtime
        import detection as _self
        _self.MODEL_PATH = args.model

    run(
        source=args.source if args.source != "0" else 0,
        show=not args.no_show,
        conf_threshold=args.conf,
        debug=args.debug,
    )
