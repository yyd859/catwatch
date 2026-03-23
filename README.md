# 🐱 CatWatch

实时追踪猫在家里哪个区域，通过 API 对外提供查询。

## 系统架构

```
摄像头/视频
    ↓
detection.py  (YOLOv8 追踪)
    ↓
zones.py      (像素坐标 → 区域名称)
    ↓
logger.py     (写入 SQLite)
    ↓
inference.py  (时间衰减推断)
    ↓
api.py        (FastAPI HTTP 接口)
    ↓
OpenClaw      (回答"猫在哪里")
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置区域（重要！）

编辑 `zones.py`，把 `ZONES` 字典改成你家的实际区域坐标。

坐标获取方法：
1. 用图片查看器打开地图（`map/` 目录下）
2. 鼠标悬停找到各区域角点的 (x, y) 坐标
3. 填入对应区域的多边形顶点

### 3. 测试检测（用视频文件）

```bash
# 用视频测试
python detection.py --source cat_video.mp4

# 用图片快速验证
python detection.py --source cat.jpg

# 树莓派摄像头（无头模式）
python detection.py --no-show
```

### 4. 启动 API

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

### 5. 查询猫的位置

```bash
curl http://localhost:8080/where_is_cat
```

返回：
```json
{
  "location": "沙发",
  "confidence": "high",
  "answer": "🐱 猫在沙发，12秒前看到的（置信度：高）"
}
```

## 文件结构

```
catwatch/
├── detection.py    # YOLO 检测主循环
├── zones.py        # 区域定义 & 坐标映射
├── logger.py       # SQLite 日志
├── inference.py    # 位置推断（时间衰减）
├── api.py          # FastAPI 接口
├── requirements.txt
├── map/            # 放地图图片（扫地机器人导出的）
├── data/           # SQLite 数据库（自动创建）
└── logs/           # 预留日志目录
```

## 树莓派部署

到货后：
1. `git clone` 或复制这个目录到树莓派
2. `pip install -r requirements.txt`
3. 接摄像头，`python detection.py --no-show`
4. `uvicorn api:app --host 0.0.0.0 --port 8080`
5. OpenClaw 配置调用 `http://<树莓派IP>:8080/where_is_cat`

## OpenClaw 集成

```
"猫在哪里？" → GET /where_is_cat → "🐱 猫在沙发，12秒前看到的"
```
