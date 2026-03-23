"""
visualize.py — 可视化猫的检测结果
在每一帧上画出红色边框 + 区域标签，输出为视频文件

用法：
  python visualize.py --source cat.mp4
  python visualize.py --source cat.mp4 --output output.mp4 --conf 0.25
"""

import argparse
import cv2
import time
from ultralytics import YOLO
from zones import map_to_zone, ZONES

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16  # 黑白花猫易被误识别为狗
PET_CLASS_IDS = [CAT_CLASS_ID, DOG_CLASS_ID]
MODEL_PATH = "yolov8m.pt"


def draw_zones(frame):
    """在画面上半透明地画出所有区域边界（灰色虚线）"""
    overlay = frame.copy()
    for zone_name, polygon in ZONES.items():
        pts = [[[x, y]] for x, y in polygon]
        import numpy as np
        pts = np.array([[x, y] for x, y in polygon], dtype=int)
        cv2.polylines(overlay, [pts], isClosed=True, color=(180, 180, 180), thickness=1, lineType=cv2.LINE_AA)
        # 区域名称标在中心
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(overlay, zone_name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    # 半透明叠加
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    return frame


def run(source, output, conf_threshold, show_zones):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频：{source}")
        return

    # 读取视频参数
    fps = cap.read() and cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    fps = fps if fps > 0 else 30

    # 跑一次 track，收集所有帧的检测结果
    print(f"🐱 正在处理视频：{source}")
    print(f"   置信度阈值：{conf_threshold}")

    # 先跑 track 收集结果
    results_list = list(model.track(
        source=source,
        stream=True,
        classes=PET_CLASS_IDS,
        conf=conf_threshold,
        show=False,
        verbose=False,
    ))

    if not results_list:
        print("❌ 没有检测到任何帧")
        return

    # 用第一帧获取分辨率
    h, w = results_list[0].orig_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    detected_count = 0
    total_frames = len(results_list)

    for i, result in enumerate(results_list):
        frame = result.orig_img.copy()

        # 画区域边界（可选）
        if show_zones:
            frame = draw_zones(frame)

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                zone = map_to_zone(cx, cy)

                # 红色边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 顶部标签背景
                label = f"Cat #{track_id} | {zone} | {conf:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 中心点
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)

                detected_count += 1

        # 帧计数水印
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"\n✅ 完成！共 {total_frames} 帧，检测到猫 {detected_count} 次")
    print(f"📹 输出视频：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="猫检测可视化")
    parser.add_argument("--source", default="cat.mp4", help="输入视频路径")
    parser.add_argument("--output", default="output.mp4", help="输出视频路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值（默认0.25）")
    parser.add_argument("--no-zones", action="store_true", help="不显示区域边界")
    args = parser.parse_args()

    run(
        source=args.source,
        output=args.output,
        conf_threshold=args.conf,
        show_zones=not args.no_zones,
    )
