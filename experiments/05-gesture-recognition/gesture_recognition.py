import gc
import time

import image
import lcd
import sensor
from maix import KPU


# 模型文件复制到 TF/SD 卡根目录后使用此路径。
MODEL_PATH = "/sd/det.kmodel"

# 类别顺序来自训练产物 label.txt，不能随意调整。
LABELS = (
    "digit_0",
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
)

# anchor.txt 第一行是像素 anchors，第二行是 YOLOv2 解码需要的
# 归一化 anchors。这里严格采用第二行数据。
ANCHORS = (
    4.45, 5.45,
    5.09, 6.44,
    6.00, 5.93,
    6.91, 6.58,
    8.41, 6.78,
)

# det.py 中给出的模型参数：输入 320x240，输出网格 10x8。
IMAGE_W = 320
IMAGE_H = 240
NET_W = 320
NET_H = 240
LAYER_W = 10
LAYER_H = 8

DETECT_THRESHOLD = 0.50
NMS_THRESHOLD = 0.30

# 对最近若干帧做类别投票，降低数字在相邻类别之间跳变的概率。
VOTE_WINDOW = 5
MIN_VOTES = 3


def append_vote(history, class_id):
    history.append(class_id)
    if len(history) > VOTE_WINDOW:
        history.pop(0)


def stable_class(history):
    """返回票数足够的类别；结果尚未稳定时返回 -1。"""
    best_class = -1
    best_count = 0

    for candidate in history:
        if candidate < 0:
            continue
        count = 0
        for value in history:
            if value == candidate:
                count += 1
        if count > best_count:
            best_class = candidate
            best_count = count

    if best_count >= MIN_VOTES:
        return best_class
    return -1


def highest_confidence_detection(detections):
    """单手场景只取最高置信度框，避免同一只手出现多个框。"""
    best = None
    if detections:
        for detection in detections:
            class_id = int(detection[4])
            if class_id < 0 or class_id >= len(LABELS):
                continue
            if best is None or detection[5] > best[5]:
                best = detection
    return best


lcd.init()
# 【关键】：若屏幕画面上下倒了，改这里的 rotation(0 或 2)，千万不要改 sensor！
# 旋转 LCD 屏幕只会改变人眼看到的画面，绝不会破坏送入 AI 模型的像素方向。
lcd.rotation(2)
lcd.clear(lcd.BLACK)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)

# 【核心】：模型 det.kmodel 是在 vflip=True, hmirror=True 下训练导出的。
# 必须保持 True，否则送进 KPU 的图像会上下倒立，导致手势特征完全错乱、只能认出 5。
sensor.set_vflip(True)
sensor.set_hmirror(True)
sensor.skip_frames(time=1000)

clock = time.clock()
gesture_kpu = KPU()
history = []

try:
    print("loading gesture model:", MODEL_PATH)
    gesture_kpu.load_kmodel(MODEL_PATH)
    gesture_kpu.init_yolo2(
        ANCHORS,
        anchor_num=len(ANCHORS) // 2,
        img_w=IMAGE_W,
        img_h=IMAGE_H,
        net_w=NET_W,
        net_h=NET_H,
        layer_w=LAYER_W,
        layer_h=LAYER_H,
        threshold=DETECT_THRESHOLD,
        nms_value=NMS_THRESHOLD,
        classes=len(LABELS),
    )
    print("gesture model ready")

    while True:
        gc.collect()
        clock.tick()
        img = sensor.snapshot()

        # det.kmodel 的网络输入就是 320x240 RGB565，因此直接送入 KPU，
        # 不再额外裁剪或拉伸，确保预处理与训练导出的 det.py 一致。
        gesture_kpu.run_with_output(img)
        detections = gesture_kpu.regionlayer_yolo2()
        best = highest_confidence_detection(detections)

        if best is None:
            append_vote(history, -1)
        else:
            append_vote(history, int(best[4]))

        voted_id = stable_class(history)

        if best is not None:
            x = int(best[0])
            y = int(best[1])
            w = int(best[2])
            h = int(best[3])
            raw_id = int(best[4])
            confidence = float(best[5])

            img.draw_rectangle(x, y, w, h, color=(0, 255, 0))

            if voted_id >= 0:
                text = "%s %.2f" % (LABELS[voted_id], confidence)
                text_color = (0, 255, 0)
            else:
                # 前几帧尚未形成多数票时显示当前结果，前缀 ? 表示待确认。
                text = "?%s %.2f" % (LABELS[raw_id], confidence)
                text_color = (255, 255, 0)

            text_y = y - 18
            if text_y < 0:
                text_y = y + 2
            img.draw_string(x, text_y, text, color=text_color, scale=1.5)
            print(text)
        else:
            img.draw_string(4, 4, "gesture: none",
                            color=(255, 0, 0), scale=1.5)

        img.draw_string(4, 220, "%2.1f fps" % clock.fps(),
                        color=(0, 60, 255), scale=1.5)
        lcd.display(img)

except Exception as err:
    print("gesture recognition error:", err)

finally:
    gesture_kpu.deinit()
    del gesture_kpu
    gc.collect()
