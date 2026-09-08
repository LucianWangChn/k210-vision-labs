# ==============================================================================
# K210 硬件按键高可靠检测与诊断程序 (工业级状态机 + 双向滞后防抖)
# 解决痛点：长按误计两次、抖动误触、边沿模式选择 (上升沿/下降沿)
# 适用硬件：亚博智能 K210 / Sipeed Maix 系列 / CanMV 各类开发板
# ==============================================================================
import time
from fpioa_manager import fm
try:
    from Maix import GPIO
except ImportError:
    from maix import GPIO
import lcd
import image

# ==================== 用户核心配置 ====================
# 触发边沿模式设置：
# "FALLING": 下降沿触发 (按下瞬间立即计数，快门首选，响应零等待)
# "RISING" : 上升沿触发 (松手瞬间才计数，适合按键释放确认)
TRIGGER_MODE = "FALLING"

# 硬件双向防抖时间 (毫秒)：
# 机械金属弹片在按下和松开时会产生 10~30ms 的剧烈高频毛刺，
# 设置为 35ms 可 100% 滤除毛刺与长按中的接触阻抗波动。
DEBOUNCE_MS = 35
# ====================================================

# ----------------- 1. 初始化 LCD 屏幕 -----------------
lcd_ready = False
try:
    lcd.init()
    lcd.clear(lcd.BLACK)
    lcd_ready = True
except Exception:
    pass

# ----------------- 2. 初始化板载 RGB 绿灯 (IO26) -----------------
led_green = None
try:
    fm.register(26, fm.fpioa.GPIOHS2)
    led_green = GPIO(GPIO.GPIOHS2, GPIO.OUT, value=1)  # 1 为熄灭
except Exception:
    pass

def set_led(on):
    if led_green:
        led_green.value(0 if on else 1)

# ----------------- 3. 工业级防抖按键检测类 -----------------
class RobustKeyDetector:
    """
    具备严格锁定机制的按键检测器：
    - 一次按下绝对只触发一次计数，无论按住 1 秒还是 10 分钟，绝不会二次误计！
    - 针对按下（下降沿）与松开（上升沿）提供对称时间窗口滤波，彻底消灭触点颤动。
    """
    def __init__(self, name, pin, hs_idx, trigger_mode="FALLING", debounce_ms=35):
        self.name = name
        self.pin = pin
        self.hs_signal = getattr(fm.fpioa, "GPIOHS%d" % hs_idx)
        self.hs_enum = getattr(GPIO, "GPIOHS%d" % hs_idx)
        
        fm.register(self.pin, self.hs_signal)
        # 上拉输入：平时未按下为 1 (3.3V)，按下导通到 GND 为 0
        self.gpio = GPIO(self.hs_enum, GPIO.IN, GPIO.PULL_UP)
        
        self.trigger_mode = trigger_mode
        self.debounce_ms = debounce_ms
        
        self.state = 0            # 0: 松开状态 (RELEASED), 1: 按住锁定状态 (LOCKED_PRESSED)
        self.filter_start = time.ticks_ms()
        self.press_start_time = 0
        self.hold_ms = 0
        self.press_count = 0
        self.just_triggered = False

    def update(self, now):
        # 实时采样引脚电平：0 = 导通接地(按下), 1 = 上拉高电平(松开)
        raw_val = self.gpio.value()
        self.just_triggered = False

        if self.state == 0:
            # 当前处于【松开状态】：监听按下信号 (0)
            if raw_val == 0:
                # 只有低电平持续稳定超过防抖时间，才确认为有效按下 (下降沿生效)
                if time.ticks_diff(now, self.filter_start) >= self.debounce_ms:
                    self.state = 1  # 进入【锁定状态】
                    self.press_start_time = now
                    self.hold_ms = 0
                    if self.trigger_mode == "FALLING":
                        self.press_count += 1
                        self.just_triggered = True
            else:
                self.filter_start = now  # 出现高电平立即重置滤波计时
        else:
            # 当前处于【按住锁定状态】：无论保持多久，绝不重复计数！
            self.hold_ms = time.ticks_diff(now, self.press_start_time)
            if raw_val == 1:
                # 只有高电平持续稳定超过防抖时间，才确认为有效松开 (上升沿生效)
                if time.ticks_diff(now, self.filter_start) >= self.debounce_ms:
                    self.state = 0  # 恢复到【松开状态】
                    if self.trigger_mode == "RISING":
                        self.press_count += 1
                        self.just_triggered = True
            else:
                self.filter_start = now  # 依然保持按下时重置松开滤波计时

# ----------------- 4. 注册按键对象 -----------------
# 亚博开发板按键 K1: IO16
detector_k1 = RobustKeyDetector("K1", 16, 0, trigger_mode=TRIGGER_MODE, debounce_ms=DEBOUNCE_MS)

# 通用 BOOT 按键: IO0
boot_pin = 0
try:
    from board import board_info
    if hasattr(board_info, 'BOOT_KEY'):
        boot_pin = board_info.BOOT_KEY
except Exception:
    pass
detector_boot = RobustKeyDetector("BOOT", boot_pin, 1, trigger_mode=TRIGGER_MODE, debounce_ms=DEBOUNCE_MS)

print("=" * 65)
print("===   K210 高可靠按键诊断系统已启动 (抗抖防重复锁定版)   ===")
print("  * 触发模式: %s沿触发 (%s)" % (
    "下降" if TRIGGER_MODE == "FALLING" else "上升",
    "按下瞬间响应" if TRIGGER_MODE == "FALLING" else "松手瞬间响应"))
print("  * 防抖滤波: %d ms (对称滞后滤波，彻底解决长按二次计数)" % DEBOUNCE_MS)
print("  * 监听引脚: K1 -> IO16 | BOOT -> IO%d" % boot_pin)
print("=" * 65)

canvas = image.Image(size=(320, 240))

try:
    while True:
        now = time.ticks_ms()
        
        detector_k1.update(now)
        detector_boot.update(now)

        # ---------- K1 触发与日志 ----------
        if detector_k1.just_triggered:
            edge_type = "下降沿 (按下)" if TRIGGER_MODE == "FALLING" else "上升沿 (松开)"
            print("[EVENT] >>> K1 (IO16) 触发！模式: %s, 累计次数: %d <<<" % (edge_type, detector_k1.press_count))
            
        # 板载 LED 随物理按住状态实时点亮
        set_led(detector_k1.state == 1)

        # ---------- BOOT 触发与日志 ----------
        if detector_boot.just_triggered:
            edge_type = "下降沿 (按下)" if TRIGGER_MODE == "FALLING" else "上升沿 (松开)"
            print("[EVENT] >>> BOOT (IO%d) 触发！模式: %s, 累计次数: %d <<<" % (boot_pin, edge_type, detector_boot.press_count))

        # ---------- LCD 屏幕动态渲染 ----------
        if lcd_ready:
            canvas.clear()
            
            # 背景色：只要有任意键处于按住状态即变色反馈
            if detector_k1.state == 1:
                canvas.draw_rectangle(0, 0, 320, 240, color=(0, 130, 40), fill=True)
                canvas.draw_string(20, 15, ">>> K1 HELD DOWN! <<<", color=(255, 255, 255), scale=2)
            elif detector_boot.state == 1:
                canvas.draw_rectangle(0, 0, 320, 240, color=(160, 90, 0), fill=True)
                canvas.draw_string(20, 15, ">>> BOOT HELD DOWN! <<<", color=(255, 255, 255), scale=2)
            else:
                canvas.draw_rectangle(0, 0, 320, 240, color=(15, 25, 45), fill=True)
                mode_str = "FALLING(Press)" if TRIGGER_MODE == "FALLING" else "RISING(Release)"
                canvas.draw_string(20, 15, "KEY TEST [%s]" % mode_str, color=(0, 220, 255), scale=2)

            canvas.draw_rectangle(5, 5, 310, 230, color=(80, 140, 220), thickness=2)

            # K1 实时状态
            if detector_k1.state == 1:
                k1_info = "LOCKED (%d ms)" % detector_k1.hold_ms
                k1_color = (0, 255, 0)
            else:
                k1_info = "RELEASED"
                k1_color = (200, 200, 200)
            canvas.draw_string(20, 55, "K1 (IO16): %s" % k1_info, color=k1_color, scale=2)
            canvas.draw_string(20, 85, "K1 Count : %d times" % detector_k1.press_count, color=(255, 255, 0), scale=2)

            # BOOT 实时状态
            if detector_boot.state == 1:
                boot_info = "LOCKED (%d ms)" % detector_boot.hold_ms
                boot_color = (255, 200, 0)
            else:
                boot_info = "RELEASED"
                boot_color = (170, 170, 170)
            canvas.draw_string(20, 125, "BOOT(IO%d): %s" % (boot_pin, boot_info), color=boot_color, scale=2)
            canvas.draw_string(20, 155, "BOOT Count: %d times" % detector_boot.press_count, color=(255, 255, 0), scale=2)

            # 底部提示
            canvas.draw_string(20, 195, "Debounce Filter: %d ms" % DEBOUNCE_MS, color=(180, 180, 220), scale=1)
            canvas.draw_string(20, 212, "Long press won't multi-count!", color=(100, 255, 100), scale=1)

            lcd.display(canvas)

        time.sleep_ms(15)

except KeyboardInterrupt:
    print("\n[INFO] 程序已退出。")
finally:
    set_led(False)
    if lcd_ready:
        lcd.clear(lcd.BLACK)
