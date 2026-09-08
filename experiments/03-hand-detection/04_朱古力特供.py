import lcd, sensor, image, gc, time, os
from fpioa_manager import fm
from maix import GPIO, KPU

# 高可靠双按键检测类 (同时支持 K1 与 BOOT，开机防抖防误触)
class DualKeyTrigger:
    def __init__(self, key_configs=[("K1", 16, 0), ("BOOT", 0, 1)]):
        self.keys = []
        self.init_time = time.ticks_ms()
        self.state = 0
        for name, pin, hs_idx in key_configs:
            try:
                hs_signal = getattr(fm.fpioa, "GPIOHS%d" % hs_idx)
                hs_enum = getattr(GPIO, "GPIOHS%d" % hs_idx)
                fm.register(pin, hs_signal)
                gpio = GPIO(hs_enum, GPIO.IN, GPIO.PULL_UP)
                self.keys.append((name, gpio))
            except Exception as e:
                pass

    def check_pressed(self):
        # 开机前 1.5 秒物理屏蔽，杜绝串口 DTR 复位瞬间的误触发
        if time.ticks_diff(time.ticks_ms(), self.init_time) < 1500:
            return False
        
        # 只要任意一个按键被按下 (电平为0)，即判定为按下
        any_down = False
        for name, gpio in self.keys:
            if gpio.value() == 0:
                any_down = True
                break
                
        if self.state == 0:
            if any_down:
                self.state = 1
                return True
        else:
            if not any_down: # 全部松开才解除锁定
                self.state = 0
        return False

detector_key = DualKeyTrigger([("K1", 16, 0), ("BOOT", 0, 1)])

save_dir = "/sd/hand_dataset"
try:
    os.mkdir(save_dir)
except Exception:
    pass

snap_id = 1
try:
    files = os.listdir(save_dir)
    existing = [int(f.replace("hand_", "").replace(".jpg", "")) for f in files if f.startswith("hand_") and f.endswith(".jpg") and f.replace("hand_", "").replace(".jpg", "").isdigit()]
    if existing:
        snap_id = max(existing) + 1
except Exception:
    pass

# K210 YOLOv2 hand detector - improved real-time version.
MODEL_PATH = "/sd/KPU/yolo_hand_detect/hand_detect.kmodel"

# 0.68 in the reference code is too strict for fists. 0.45 is a practical
# starting point. If false positives remain, try 0.48 or 0.50.
DETECT_THRESHOLD = 0.45

# A lower IoU threshold makes NMS suppress overlapping duplicate boxes earlier.
KPU_NMS_THRESHOLD = 0.20
SOFT_NMS_THRESHOLD = 0.18

# The current project is a single-hand interaction. Keeping one result removes
# the occasional double box completely. Set this to 2 only when two real hands
# must be detected at the same time.
MAX_HANDS = 1

# Small or extremely thin boxes are usually false detections. A fist is close
# to square and is not removed by these limits.
MIN_BOX_AREA = 18 * 18
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0

# Tracking parameters: two consistent frames confirm a new hand; an established
# box may survive one missed frame, which reduces flicker without a long ghost box.
NEW_TARGET_CONFIRM_FRAMES = 2
MAX_MISSED_FRAMES = 1
SMOOTH_ALPHA = 0.65

anchor = (0.8125, 0.4556, 1.1328, 1.2667, 1.8594,
          1.4889, 1.4844, 2.2000, 2.6484, 2.9333)
names = ['hand']


def box_iou(a, b):
    """IoU for [x, y, w, h, class_id, confidence]."""
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return float(inter) / union


def valid_box(box):
    """Remove common background false positives without excluding fists."""
    w = box[2]
    h = box[3]
    if w <= 0 or h <= 0 or w * h < MIN_BOX_AREA:
        return False
    ratio = float(w) / h
    return MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO


def software_nms(raw_hands):
    """Second NMS pass; KPU results are converted to ordinary lists."""
    candidates = []
    if raw_hands:
        for hand in raw_hands:
            box = [int(hand[0]), int(hand[1]), int(hand[2]), int(hand[3]),
                   int(hand[4]), float(hand[5])]
            if valid_box(box):
                candidates.append(box)

    candidates.sort(key=lambda item: item[5], reverse=True)
    kept = []
    for box in candidates:
        duplicate = False
        for selected in kept:
            if box_iou(box, selected) >= SOFT_NMS_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
            if len(kept) >= MAX_HANDS:
                break
    return kept


def smooth_box(new_box, old_box):
    """Smooth a continuing target but never drag a box across the screen."""
    if old_box is None or box_iou(new_box, old_box) < 0.10:
        return new_box

    alpha = SMOOTH_ALPHA
    beta = 1.0 - alpha
    return [
        int(new_box[0] * alpha + old_box[0] * beta),
        int(new_box[1] * alpha + old_box[1] * beta),
        int(new_box[2] * alpha + old_box[2] * beta),
        int(new_box[3] * alpha + old_box[3] * beta),
        new_box[4],
        new_box[5]
    ]


lcd.init()
sensor.reset()
sensor.set_framesize(sensor.QVGA)
sensor.set_pixformat(sensor.RGB565)
# Give automatic exposure/white balance time to settle before detection.
sensor.skip_frames(time=1500)

# Uncomment when the camera orientation requires it.
# sensor.set_vflip(True)

resize_img = image.Image(size=(320, 256))
hand_detector = KPU()
hand_detector.load_kmodel(MODEL_PATH)
hand_detector.init_yolo2(
    anchor,
    anchor_num=len(anchor) // 2,
    img_w=320,
    img_h=240,
    net_w=320,
    net_h=256,
    layer_w=10,
    layer_h=8,
    threshold=DETECT_THRESHOLD,
    nms_value=KPU_NMS_THRESHOLD,
    classes=len(names)
)

tracked_box = None
pending_box = None
pending_frames = 0
missed_frames = 0

try:
    while True:
        gc.collect()
        img = sensor.snapshot()

        # Keep the model's original preprocessing path, as required by the
        # supplied kmodel: QVGA camera image in a 320 x 256 KPU input buffer.
        resize_img.draw_image(img, 0, 0).pix_to_ai()
        hand_detector.run_with_output(resize_img)
        candidates = software_nms(hand_detector.regionlayer_yolo2())

        if candidates:
            candidate = candidates[0]

            if tracked_box is not None and box_iou(candidate, tracked_box) >= 0.10:
                tracked_box = smooth_box(candidate, tracked_box)
                pending_box = None
                pending_frames = 0
                missed_frames = 0
            elif pending_box is not None and box_iou(candidate, pending_box) >= 0.08:
                pending_box = smooth_box(candidate, pending_box)
                pending_frames += 1
                if pending_frames >= NEW_TARGET_CONFIRM_FRAMES:
                    tracked_box = pending_box
                    pending_box = None
                    pending_frames = 0
                    missed_frames = 0
                elif tracked_box is not None:
                    missed_frames += 1
            else:
                pending_box = candidate
                pending_frames = 1
                if tracked_box is not None:
                    missed_frames += 1

            if tracked_box is not None and missed_frames > MAX_MISSED_FRAMES:
                tracked_box = None
        else:
            pending_box = None
            pending_frames = 0
            if tracked_box is not None:
                missed_frames += 1
                if missed_frames > MAX_MISSED_FRAMES:
                    tracked_box = None

        if tracked_box is not None:
            x, y, w, h = tracked_box[0], tracked_box[1], tracked_box[2], tracked_box[3]
            class_id = tracked_box[4]
            confidence = tracked_box[5]
            img.draw_rectangle(x, y, w, h, color=(0, 255, 0), thickness=2)
            img.draw_string(x + 2, y + 2, "%.2f" % confidence, color=(0, 255, 0))
            img.draw_string(x + 2, y + 12, names[class_id], color=(0, 255, 0))

        img.draw_string(10, 10, "K1/BOOT: Snap Hand", color=(0, 255, 255), scale=1)

        # 检测快门按键 (支持 K1 与 BOOT)
        if detector_key.check_pressed():
            save_path = "%s/hand_%03d.jpg" % (save_dir, snap_id)
            img.save(save_path)
            print("[实时快门] 按键触发！已抓拍保存: %s" % save_path)
            img.draw_rectangle(0, 0, 320, 240, color=(0, 255, 0), thickness=6)
            img.draw_string(20, 30, "SAVED: hand_%03d.jpg" % snap_id, color=(0, 255, 0), scale=2)
            lcd.display(img)
            snap_id += 1
            time.sleep_ms(300)
        else:
            lcd.display(img)

except Exception as err:
    print("hand detector error:", err)

finally:
    hand_detector.deinit()
    del hand_detector
    del resize_img
    gc.collect()
