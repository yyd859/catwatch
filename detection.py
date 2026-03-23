"""
detection.py — 猫检测主循环（运动辅助版）

架构：
  1. 背景减除（MOG2）找运动区域 → 触发 YOLO 推理
  2. YOLO 只在运动区域推理（省算力，过滤误检）
  3. 无运动时每 STATIC_INTERVAL 帧全图跑一次（抓住睡觉的猫）
  4. YOLO 是最终裁判：只有它说是猫/狗才记录

用法：
  python detection.py                     # 使用摄像头
  python detection.py --source cat.mp4    # 使用视频文件
  python detection.py --no-show --source cat.mp4
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

from zones import map_to_zone
from logger import init_db, log_detection, Detection

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16
PET_CLASS_IDS = [CAT_CLASS_ID, DOG_CLASS_ID]

# 模型：yolov8m 平衡精度和速度
MODEL_PATH = "yolov8m.pt"

# 无运动时每隔多少帧全图扫一次（抓住静止的猫）
STATIC_INTERVAL = 30

# 运动检测参数
MOG2_HISTORY = 500          # 背景模型学习帧数
MOG2_THRESHOLD = 50         # 像素变化阈值
MOTION_MIN_AREA = 1500      # 最小运动区域面积（过滤噪点）
MOTION_PADDING = 40         # 运动区域外扩像素（避免裁切太紧）


def get_motion_regions(fgmask, frame_w, frame_h):
    """
    从前景掩码提取运动区域，返回合并后的 bounding box 列表。
    每个 box: (x1, y1, x2, y2)
    """
    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MOTION_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # 外扩 padding，不超出边界
        x1 = max(0, x - MOTION_PADDING)
        y1 = max(0, y - MOTION_PADDING)
        x2 = min(frame_w, x + w + MOTION_PADDING)
        y2 = min(frame_h, y + h + MOTION_PADDING)
        regions.append((x1, y1, x2, y2))

    return regions


def merge_regions(regions):
    """合并重叠的运动区域，减少重复推理。"""
    if not regions:
        return []
    merged = list(regions)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                ax1, ay1, ax2, ay2 = merged[j]
                # 有重叠则合并
                if x1 < ax2 and x2 > ax1 and y1 < ay2 and y2 > ay1:
                    x1 = min(x1, ax1)
                    y1 = min(y1, ay1)
                    x2 = max(x2, ax2)
                    y2 = max(y2, ay2)
                    used[j] = True
                    changed = True
            result.append((x1, y1, x2, y2))
            used[i] = True
        merged = result
    return merged


def run_yolo_on_regions(model, frame, regions, conf_threshold):
    """
    在指定区域裁剪图像跑 YOLO，返回检测结果列表。
    每个结果: (track_id, zone, conf, cx, cy, x1, y1, x2, y2)
    注意：坐标已转换回原图坐标系。
    """
    detections = []
    h, w = frame.shape[:2]

    for (rx1, ry1, rx2, ry2) in regions:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        results = model.predict(
            crop,
            classes=PET_CLASS_IDS,
            conf=conf_threshold,
            verbose=False,
        )
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                # 坐标转回原图
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                ox1 = int(bx1) + rx1
                oy1 = int(by1) + ry1
                ox2 = int(bx2) + rx1
                oy2 = int(by2) + ry1
                cx = (ox1 + ox2) / 2
                cy = (oy1 + oy2) / 2
                zone = map_to_zone(cx, cy)
                track_id = -1  # 裁剪推理没有 track id
                detections.append((track_id, zone, conf, cx, cy, ox1, oy1, ox2, oy2))

    return detections


def run_yolo_full_frame(model, frame, conf_threshold):
    """全图跑 YOLO track（用于静止场景兜底）。"""
    detections = []
    results = model.track(
        frame,
        classes=PET_CLASS_IDS,
        conf=conf_threshold,
        verbose=False,
        persist=True,
    )
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            zone = map_to_zone(cx, cy)
            detections.append((track_id, zone, conf, cx, cy,
                                int(x1), int(y1), int(x2), int(y2)))
    return detections


def run(source=0, show=True, conf_threshold=0.15, debug=False):
    init_db()
    model = YOLO(MODEL_PATH)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_THRESHOLD,
        detectShadows=False,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源：{source}")
        return

    print(f"✅ 模型：{MODEL_PATH}")
    print(f"📹 视频源：{source}")
    print(f"⚙️  置信度阈值：{conf_threshold}")
    print("🐱 开始检测（运动辅助模式）...\n")

    frame_idx = 0
    total_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]
        ts = time.time()

        # ── 1. 背景减除，提取运动掩码 ──
        fgmask = bg_subtractor.apply(frame)
        motion_regions = get_motion_regions(fgmask, w, h)
        motion_regions = merge_regions(motion_regions)
        has_motion = len(motion_regions) > 0

        # ── 2. 决定推理策略 ──
        detections = []
        mode = ""

        if has_motion:
            # 有运动：只在运动区域推理
            detections = run_yolo_on_regions(model, frame, motion_regions, conf_threshold)
            mode = f"motion({len(motion_regions)} regions)"
        elif frame_idx % STATIC_INTERVAL == 0:
            # 无运动：定时全图扫（抓住静止的猫）
            detections = run_yolo_full_frame(model, frame, conf_threshold)
            mode = "static_scan"

        # ── 3. 记录检测结果 ──
        for (track_id, zone, conf, cx, cy, x1, y1, x2, y2) in detections:
            det = Detection(
                timestamp=ts,
                track_id=track_id,
                zone=zone,
                confidence=conf,
                cx=cx,
                cy=cy,
            )
            log_detection(det)
            total_detections += 1
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"🐱 ID={track_id:3d} | "
                f"区域={zone:8s} | "
                f"置信度={conf:.2f} | "
                f"位置=({cx:.0f},{cy:.0f}) | "
                f"模式={mode}"
            )

        # ── 4. 可视化（可选）──
        if show:
            vis = frame.copy()

            # 画运动区域（蓝色虚线框）
            if debug and has_motion:
                for (rx1, ry1, rx2, ry2) in motion_regions:
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 100, 0), 1)

            # 画猫的检测框（红色）
            for (track_id, zone, conf, cx, cy, x1, y1, x2, y2) in detections:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Cat #{track_id} | {zone} | {conf:.2f}"
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)

            # 状态水印
            status = f"Frame {frame_idx} | {'MOTION' if has_motion else 'static'} | cats: {total_detections}"
            cv2.putText(vis, status, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("CatWatch", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
    print(f"\n✅ 结束。共检测到猫 {total_detections} 次（{frame_idx} 帧）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猫位置检测（运动辅助版）")
    parser.add_argument("--source", default=0,
                        help="视频源（0=摄像头，或视频/图片路径）")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示实时画面（无头模式，树莓派用）")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="置信度阈值（默认0.15）")
    parser.add_argument("--debug", action="store_true",
                        help="显示运动区域蓝框（调试用）")
    args = parser.parse_args()

    run(
        source=args.source if args.source != "0" else 0,
        show=not args.no_show,
        conf_threshold=args.conf,
        debug=args.debug,
    )
