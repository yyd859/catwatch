"""
visualize.py — 可视化猫的检测结果（运动辅助版）

用法：
  python visualize.py --source cat.mp4
  python visualize.py --source cat.mp4 --output output.mp4 --conf 0.15
  python visualize.py --source cat.mp4 --debug   # 显示运动区域蓝框
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from zones import map_to_zone, ZONES

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16
PET_CLASS_IDS = [CAT_CLASS_ID, DOG_CLASS_ID]
MODEL_PATH = "yolov8m.pt"

STATIC_INTERVAL = 30
MOG2_HISTORY = 500
MOG2_THRESHOLD = 50
MOTION_MIN_AREA = 1500
MOTION_PADDING = 40


def get_motion_regions(fgmask, w, h):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MOTION_MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        regions.append((max(0, x - MOTION_PADDING), max(0, y - MOTION_PADDING),
                        min(w, x + bw + MOTION_PADDING), min(h, y + bh + MOTION_PADDING)))
    return regions


def merge_regions(regions):
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
                if x1 < ax2 and x2 > ax1 and y1 < ay2 and y2 > ay1:
                    x1, y1 = min(x1, ax1), min(y1, ay1)
                    x2, y2 = max(x2, ax2), max(y2, ay2)
                    used[j] = True
                    changed = True
            result.append((x1, y1, x2, y2))
            used[i] = True
        merged = result
    return merged


def draw_zones(frame):
    overlay = frame.copy()
    for zone_name, polygon in ZONES.items():
        pts = np.array([[x, y] for x, y in polygon], dtype=int)
        cv2.polylines(overlay, [pts], isClosed=True, color=(180, 180, 180), thickness=1)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(overlay, zone_name, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    return frame


def run_yolo_on_regions(model, frame, regions, conf_threshold):
    detections = []
    for (rx1, ry1, rx2, ry2) in regions:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue
        results = model.predict(crop, classes=PET_CLASS_IDS, conf=conf_threshold, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                ox1, oy1 = int(bx1) + rx1, int(by1) + ry1
                ox2, oy2 = int(bx2) + rx1, int(by2) + ry1
                cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
                detections.append((-1, map_to_zone(cx, cy), conf, cx, cy, ox1, oy1, ox2, oy2))
    return detections


def run_yolo_full(model, frame, conf_threshold):
    detections = []
    results = model.predict(frame, classes=PET_CLASS_IDS, conf=conf_threshold, verbose=False)
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((-1, map_to_zone(cx, cy), conf, cx, cy,
                                int(x1), int(y1), int(x2), int(y2)))
    return detections


def run(source, output, conf_threshold, show_zones, debug):
    model = YOLO(MODEL_PATH)
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_THRESHOLD, detectShadows=False)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    if not ret:
        print("❌ 读取第一帧失败")
        return
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    print(f"🐱 处理中：{source}  ({total_frames} 帧)")
    detected_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        fgmask = bg_sub.apply(frame)
        motion_regions = merge_regions(get_motion_regions(fgmask, w, h))
        has_motion = len(motion_regions) > 0

        if has_motion:
            detections = run_yolo_on_regions(model, frame, motion_regions, conf_threshold)
            mode = "motion"
        elif frame_idx % STATIC_INTERVAL == 0:
            detections = run_yolo_full(model, frame, conf_threshold)
            mode = "static"
        else:
            detections = []
            mode = "-"

        # 画面合成
        vis = frame.copy()
        if show_zones:
            vis = draw_zones(vis)

        # 运动区域蓝框（debug）
        if debug and has_motion:
            for (rx1, ry1, rx2, ry2) in motion_regions:
                cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 100, 0), 1)

        # 猫的检测框（红色）
        for (track_id, zone, conf, cx, cy, x1, y1, x2, y2) in detections:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Cat | {zone} | {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 0, 255), -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            detected_count += 1

        # 状态水印
        status = f"Frame {frame_idx}/{total_frames} | {mode} | cats: {detected_count}"
        cv2.putText(vis, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        writer.write(vis)

        if frame_idx % 100 == 0:
            print(f"  进度 {frame_idx}/{total_frames}...")

    cap.release()
    writer.release()
    print(f"\n✅ 完成！共 {frame_idx} 帧，检测到猫 {detected_count} 次")
    print(f"📹 输出：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猫检测可视化（运动辅助版）")
    parser.add_argument("--source", default="cat.mp4")
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--no-zones", action="store_true")
    parser.add_argument("--debug", action="store_true", help="显示运动区域蓝框")
    args = parser.parse_args()

    run(source=args.source, output=args.output,
        conf_threshold=args.conf,
        show_zones=not args.no_zones,
        debug=args.debug)
