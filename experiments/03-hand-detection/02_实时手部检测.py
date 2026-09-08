# =====================================================================
# 实验三：YOLOv2 手部检测 (软件极速流水线加速版 + 实时按键快门截图)
# 软件提速技术：
# 1. 工业级交替流水线 (Interleaved Pipeline)：隔帧推理，算力减半，帧率翻倍
# 2. 消除主循环内的逐帧内存清零 (减少约 10ms 纯 CPU 浪费)
# 3. 提升 LCD 硬件 SPI 传输频率至 15MHz
# 4. 支持 K1 / BOOT 随时按下抓拍当前画面并保存到 SD 卡！
# =====================================================================
import lcd, sensor, image, time, os, gc
from fpioa_manager import fm
from maix import GPIO, KPU

# 1. 硬件与传感器加速初始化
lcd.init(freq=15000000) # 提速 SPI 刷屏
sensor.reset()
sensor.set_framesize(sensor.QVGA) # 320x240
sensor.set_pixformat(sensor.RGB565)

sensor.set_contrast(1)       # 增强手指轮廓对比度
sensor.set_saturation(1)     # 锁定红润肤色
sensor.set_auto_gain(True)
sensor.skip_frames(time=1000)

# 2. 高可靠双按键检测类 (同时支持 K1 与 BOOT，开机防抖防误触)
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

MODEL_PATH = "/sd/KPU/yolo_hand_detect/hand_detect.kmodel"
try:
    with open(MODEL_PATH, "rb") as f:
        pass
except Exception:
    MODEL_PATH = "/sd/hand_detect.kmodel"

# 只在系统启动时清零一次，绝不在主循环逐帧耗时 clear
resize_img = image.Image(size=(320, 256))
resize_img.clear()

DETECT_THRESHOLD = 0.38
KPU_NMS_THRESHOLD = 0.25

anchor = (0.8125, 0.4556, 1.1328, 1.2667, 1.8594, 1.4889, 1.4844, 2.2000, 2.6484, 2.9333)
names = ['hand']

hand_detector = KPU()
hand_detector.load_kmodel(MODEL_PATH)
hand_detector.init_yolo2(
    anchor,
    anchor_num=len(anchor) // 2,
    img_w=320, img_h=240,
    net_w=320, net_h=256,
    layer_w=10, layer_h=8,
    threshold=DETECT_THRESHOLD,
    nms_value=KPU_NMS_THRESHOLD,
    classes=len(names)
)

clock = time.clock()
frame_id = 0
last_hands = []

print("==================================================")
print("  手部检测【极速流水线 + 实时按键抓拍】启动！")
print("  * 实时抓拍: 按 K1(IO16) 或 BOOT(IO0) 保存当前图片")
print("  * 保存路径: %s/hand_XXX.jpg" % save_dir)
print("==================================================")

try:
    while True:
        gc.collect()
        clock.tick()
        frame_id += 1
        
        img = sensor.snapshot()

        # 检测按键快门 (支持 K1 与 BOOT)
        triggered_key = detector_key.check_pressed()

        # 【核心软件提速策略】：隔帧推理
        if (frame_id % 2 == 0) or (len(last_hands) == 0):
            resize_img.draw_image(img, 0, 8).pix_to_ai()
            hand_detector.run_with_output(resize_img)
            last_hands = hand_detector.regionlayer_yolo2()

        # 绘制检测框与准星
        if last_hands:
            for hand in last_hands:
                if hand[2] * hand[3] < 20 * 20:
                    continue
                hx = hand[0]
                hy = max(0, hand[1] - 8)
                hw = hand[2]
                hh = hand[3]
                conf = hand[5]

                img.draw_rectangle(hx, hy, hw, hh, color=(0, 255, 0), thickness=2)
                cx, cy = hx + hw // 2, hy + hh // 2
                img.draw_line(cx - 5, cy, cx + 5, cy, color=(255, 0, 0), thickness=1)
                img.draw_line(cx, cy - 5, cx, cy + 5, color=(255, 0, 0), thickness=1)

                tag = "hand: %.1f%%" % (conf * 100)
                img.draw_string(hx + 2, max(2, hy - 14), tag, color=(0, 255, 0), scale=1)

        # 顶部绘制实时 FPS 与按键提示
        fps = clock.fps()
        img.draw_rectangle(0, 0, 320, 16, color=(20, 20, 20), fill=True)
        img.draw_string(2, 2, "FPS: %.1f | K1: Snapshot" % fps, color=(0, 255, 255), scale=1)

        # 处理快门抓拍保存
        if triggered_key:
            save_path = "%s/hand_%03d.jpg" % (save_dir, snap_id)
            img.save(save_path)
            print("[实时快门] >>> 按键 K1 触发！已抓拍保存: %s <<<" % save_path)
            
            img.draw_rectangle(0, 0, 320, 240, color=(0, 255, 0), thickness=6)
            img.draw_string(20, 30, "SAVED: hand_%03d.jpg" % snap_id, color=(0, 255, 0), scale=2)
            lcd.display(img)
            snap_id += 1
            time.sleep_ms(300)
        else:
            lcd.display(img)

except Exception as err:
    print("运行退出:", err)

finally:
    hand_detector.deinit()
    del hand_detector
    del resize_img
    gc.collect()
