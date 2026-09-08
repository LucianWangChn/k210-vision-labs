import sensor, image, lcd, time, os
from fpioa_manager import fm
from maix import GPIO

lcd.init()
lcd.clear(lcd.BLACK)

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) # 320x240
sensor.skip_frames(time=1000)

save_dir = "/sd/hand_dataset"
try:
    os.mkdir(save_dir)
except Exception:
    pass

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
                print("注册按键 %s (IO%d) 失败:" % (name, pin), e)

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

# 注册双按键快门 (支持 K1 和 BOOT 键)
detector_key = DualKeyTrigger([("K1", 16, 0), ("BOOT", 0, 1)])

photo_id = 1
try:
    files = os.listdir(save_dir)
    existing_nums = []
    for f in files:
        if f.startswith("hand_") and f.endswith(".jpg"):
            num_str = f.replace("hand_", "").replace(".jpg", "")
            if num_str.isdigit():
                existing_nums.append(int(num_str))
    if existing_nums:
        photo_id = max(existing_nums) + 1
except Exception:
    pass

print("==================================================")
print("===      手部数据采集程序 (防开机误拍版)       ===")
print("  * 支持快门按键: K1 (IO16) 或 BOOT (IO0)")
print("  * 保存路径: %s" % save_dir)
print("  * 下张编号: hand_%03d.jpg" % photo_id)
print("==================================================")

while True:
    img = sensor.snapshot()
    
    if detector_key.check_pressed():
        file_path = "%s/hand_%03d.jpg" % (save_dir, photo_id)
        img.save(file_path)
        print("[快门] 触发按键 | 已成功保存: %s" % file_path)
        
        img.draw_rectangle(0, 0, 320, 240, color=(0, 255, 0), thickness=6)
        img.draw_string(15, 15, "SAVED: hand_%03d.jpg" % photo_id, color=(0, 255, 0), scale=2)
        lcd.display(img)
        
        photo_id += 1
        time.sleep_ms(400)
    else:
        img.draw_string(10, 10, "Press K1 / BOOT to Shoot", color=(255, 255, 0), scale=1)
        img.draw_string(10, 26, "Next: hand_%03d.jpg" % photo_id, color=(0, 255, 255), scale=1)
        lcd.display(img)
