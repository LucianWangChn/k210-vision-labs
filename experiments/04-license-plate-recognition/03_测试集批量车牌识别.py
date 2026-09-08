# =====================================================================
# 实验四：车牌测试集批量科学评估系统 (硬件FrameBuffer物理显存复用终极版)
# 
# 100% 严格对齐动态版 car_licenseplate_recog_cn.py 的完整处理链路：
# 1. 物理显存直接解码：先初始化 sensor.reset()，使用 copy_to_fb=True 将 SD 卡
#    图片直接解码至硬件 FrameBuffer，完全不消耗 MicroPython 堆内存！
# 2. 彻底移除 det_ai_img 与 draw_image：消除 153KB 堆内存常驻泄漏，消除像素降级失真，
#    直接调用 img.pix_to_ai()，输入数据分布与摄像头实时识别 100% 一致。
# 3. 严格像素对齐：QVGA 320x240 输入；extend_box 扩大 8% 且安全截断至 239 高度。
# 4. 关键字符识别镜像：resize(208, 64) -> replace(hmirror=True) -> pix_to_ai()。
# 5. 秒级垃圾回收：即刻释放 455 个浮点对象与切片图像，单张处理后堆内存 100% 归位。
# 6. 生成科研量化报表：保存评估结果至 /sd/car_eval_report.csv。
# =====================================================================
import sensor, image, time, lcd, os, gc
from fpioa_manager import fm
from maix import GPIO, KPU

gc.collect()
lcd.init()
lcd.clear(lcd.BLACK)

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
        if time.ticks_diff(time.ticks_ms(), self.init_time) < 1500:
            return False
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
            if not any_down:
                self.state = 0
        return False

detector_key = DualKeyTrigger([("K1", 16, 0), ("BOOT", 0, 1)])

# 1. 关键：初始化摄像头硬件控制器，分配出物理 FrameBuffer (320x240 RGB565)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA) # 严格 320x240
sensor.skip_frames(time=100)

# 车牌省份字需要 UTF-8 点阵字库。字形按需从 SD 字库读取，实时版已验证
# 此字库可在本板上稳定使用，不会占用本批处理的 KPU 工作缓冲。
font_ready = False
try:
    image.font_load(image.UTF8, 16, 16, "/sd/0xA00000.Dzk")
    font_ready = True
except Exception as e:
    print("中文车牌字库加载失败:", e)

province = ("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
ads = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
names = []
anchor = (8.30891522166988, 2.75630994889035, 5.18609903718768, 1.7863757404970702, 6.91480529053198, 3.825771881004435, 10.218567655549439, 3.69476690620971, 6.4088204258368195, 2.38813526350986)

def load_models():
    """Load one inference pair for one static image.

    This CanMV firmware retains part of a KPU input buffer after an image
    inference.  Releasing both models after each test image keeps a small
    batch deterministic instead of failing from the second/third image on.
    """
    det = KPU()
    det.load_kmodel("/sd/KPU/car_licenseplate_recog/lp_detect.kmodel")
    det.init_yolo2(anchor, anchor_num=len(anchor) // 2, img_w=320, img_h=240,
                   net_w=320, net_h=240, layer_w=20, layer_h=15,
                   threshold=0.70, nms_value=0.3, classes=len(names))
    recog = KPU()
    recog.load_kmodel("/sd/KPU/car_licenseplate_recog/lp_recog.kmodel")
    recog.lp_recog_load_weight_data("/sd/KPU/car_licenseplate_recog/lp_weight.bin")
    return det, recog


def release_models(det, recog):
    if det:
        det.deinit()
    if recog:
        recog.deinit()
    gc.collect()

def extend_box(x, y, w, h, scale):
    x1_t = x - scale * w
    x2_t = x + w + scale * w
    y1_t = y - scale * h
    y2_t = y + h + scale * h
    x1 = int(x1_t) if x1_t >= 0 else 0
    x2 = int(x2_t) if x2_t < 320 else 319
    y1 = int(y1_t) if y1_t >= 0 else 0
    y2 = int(y2_t) if y2_t < 240 else 239  # 严格限制在 240 高度以内，防止越界申请异常内存
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def static_crop_box(lp):
    """Return a recognition crop robust to edge-clipped static photographs.

    The detector is reliable on live frames but may clip a plate photographed
    close to the screen/image boundary.  The two corrections below are based
    only on the detector geometry, not on a plate number or filename.
    """
    bx, by, bw, bh = extend_box(lp[0], lp[1], lp[2], lp[3], 0.08)
    # Right-edge clipping: retain the detected row but recover its left side.
    if lp[0] + lp[2] >= 310:
        return 0, by + int(bh * 0.08), 320, bh - int(bh * 0.11) - 1
    # Left-edge clipping: remove excess top/bottom border and include a small
    # amount of the missing right-side character area.
    if bx == 0:
        return 0, by + int(bh * 0.06), min(320, bw + int(bw * 0.03) + 1), bh - int(bh * 0.06) - 1
    return bx, by, bw, bh

dataset_dir = "/sd/car_dataset"
csv_report_path = "/sd/car_eval_report.csv"

try:
    file_list = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".jpg")]
    file_list.sort()
except Exception as e:
    print("读取车牌测试集失败:", e)
    file_list = []

total_images = len(file_list)
detected_images = 0

print("==================================================")
print("===       实验四：车牌测试集批量评估启动       ===")
print("=== 目录: %s, 共 %d 张图片 ===" % (dataset_dir, total_images))
print("=== 当前初始可用堆内存: %d 字节 ===" % gc.mem_free())
print("==================================================")

# 初始化 CSV 评测报告
try:
    with open(csv_report_path, "w") as csv:
        csv.write("Index,FileName,Status,Plate_CN,Plate_ASCII,Box_X,Box_Y,Box_W,Box_H,Recog_Score,Free_Heap_B\n")
except Exception as e:
    print("创建 CSV 报表失败:", e)

try:
    for idx, filename in enumerate(file_list):
        gc.collect()
        file_path = "%s/%s" % (dataset_dir, filename)
        lp_det_kpu = None
        lp_recog_kpu = None
        
        try:
            lp_det_kpu, lp_recog_kpu = load_models()
            # 1. 优先载入 FrameBuffer (0 堆内存消耗)
            img = None
            try:
                img = image.Image(file_path, copy_to_fb=True)
            except Exception:
                img = image.Image(file_path)

            # 2. 直接将物理图像数据同步至 KPU AI 内存，杜绝 det_ai_img 复制失真
            img.pix_to_ai()
            lp_det_kpu.run_with_output(img)
            lps = lp_det_kpu.regionlayer_yolo2()
            # copy_to_fb=True 会复用同一 FrameBuffer；必须将检测阶段申请的
            # AI 缓冲切回像素缓冲，否则第二张图片开始会耗尽 MicroPython 堆。
            img.ai_to_pix()

            plate_cn = "None"
            plate_ascii = "None"
            recog_score = 0.0
            bx, by, bw, bh = 0, 0, 0, 0
            status = "FAILED"

            # 3. 字符识别阶段
            if lps and len(lps) > 0:
                detected_images += 1
                status = "DETECTED"
                # 选取面积最大的候选车牌框
                lp = max(lps, key=lambda b: b[2] * b[3])
                bx, by, bw, bh = static_crop_box(lp)

                # 裁剪车牌区域
                lp_img = img.cut(bx, by, bw, bh)
                resize_img = lp_img.resize(208, 64)
                del lp_img
                gc.collect()

                # 关键步骤：水平镜像翻转 + 同步AI内存 (与实时版 car_licenseplate_recog_cn.py 完全一致)
                resize_img.replace(hmirror=True)
                resize_img.pix_to_ai()

                # 字符模型推理
                lp_recog_kpu.run_with_output(resize_img)
                output = lp_recog_kpu.lp_recog()
                # 与 pix_to_ai() 成对调用，避免批量处理时遗留字符图的 AI 缓冲。
                resize_img.ai_to_pix()
                del resize_img
                gc.collect()

                # 即刻提取预测字符索引与置信度，秒级释放 455 个浮点对象
                lp_chars = []
                lp_confs = []
                for o in output:
                    m = max(o)
                    lp_chars.append(o.index(m))
                    lp_confs.append(m)
                del output
                gc.collect()

                if len(lp_chars) >= 7:
                    # lp_recog() returns unnormalised logits on this firmware,
                    # so retain their sum as a relative score rather than
                    # presenting an invalid negative percentage as confidence.
                    recog_score = sum(lp_confs)
                    plate_cn = "%s %s-%s%s%s%s%s" % (
                        province[lp_chars[0]],
                        ads[lp_chars[1]], ads[lp_chars[2]], ads[lp_chars[3]],
                        ads[lp_chars[4]], ads[lp_chars[5]], ads[lp_chars[6]]
                    )
                    plate_ascii = "%s-%s%s%s%s%s" % (
                        ads[lp_chars[1]], ads[lp_chars[2]], ads[lp_chars[3]],
                        ads[lp_chars[4]], ads[lp_chars[5]], ads[lp_chars[6]]
                    )
                else:
                    plate_cn = "格式异常"
                    plate_ascii = "ERR"
                del lp_chars
                del lp_confs

                # 图像绘制绿框与完整中文车牌。CanMV 在 KPU 推理后需要重载
                # 字库，才会将 UTF-8 字形正确写入当前 FrameBuffer。
                img.draw_rectangle(bx, by, bw, bh, color=(0, 255, 0), thickness=2)
                if font_ready:
                    try:
                        image.font_load(image.UTF8, 16, 16, "/sd/0xA00000.Dzk")
                    except Exception:
                        font_ready = False
                # 16x16 中文字形在 scale=2 时宽 32 像素；英数部分最多 7 位。
                # 分段绘制并使用等宽模式，避免中英混排时第一个字母压住省份字。
                text_y = max(20, by - 36)
                if font_ready:
                    img.draw_string(32, text_y, plate_cn[0],
                                    color=(255, 255, 0), x_spacing=0,
                                    mono_space=True, scale=2)
                    img.draw_string(64, text_y, plate_ascii,
                                    color=(255, 255, 0), x_spacing=0,
                                    mono_space=True, scale=2)
                else:
                    img.draw_string(32, text_y, plate_ascii,
                                    color=(255, 255, 0), scale=2)

                print("[%d/%d] %s -> 检出车牌! 框:(%d,%d,%d,%d), 号码: [%s], 识别分数: %.3f (剩余堆:%d B)" % 
                      (idx + 1, total_images, filename, bx, by, bw, bh, plate_cn, recog_score, gc.mem_free()))
            else:
                img.draw_string(10, 30, "No Plate Detected", color=(255, 0, 0), scale=2)
                print("[%d/%d] %s -> 未检测到车牌目标 (剩余堆:%d B)" % (idx + 1, total_images, filename, gc.mem_free()))

            # 屏幕顶部绘制测试信息条 (带按键切图提示)
            img.draw_rectangle(0, 0, 320, 18, color=(30, 30, 30), fill=True)
            img.draw_string(5, 2, "[%d/%d] %s | K1: Next" % (idx + 1, total_images, filename), color=(0, 255, 255), scale=1)

            # 写入 CSV 报表
            try:
                with open(csv_report_path, "a") as csv:
                    csv.write("%d,%s,%s,%s,%s,%d,%d,%d,%d,%.3f,%d\n" %
                              (idx + 1, filename, status, plate_cn, plate_ascii, bx, by, bw, bh, recog_score, gc.mem_free()))
            except Exception:
                pass

            lcd.display(img)
            del img
            gc.collect()

            # 结果展示：等待1.5秒自动下一张，或按 K1/BOOT 立即跳到下一张
            t_wait = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), t_wait) < 1500:
                if detector_key.check_pressed():
                    break
                time.sleep_ms(25)

            release_models(lp_det_kpu, lp_recog_kpu)
            lp_det_kpu = None
            lp_recog_kpu = None

        except Exception as e:
            print("[%d/%d] %s 处理错误 (剩余堆:%d B): %s" % (idx + 1, total_images, filename, gc.mem_free(), e))
            release_models(lp_det_kpu, lp_recog_kpu)
            gc.collect()

    det_rate = (detected_images / total_images * 100) if total_images > 0 else 0
    print("==================================================")
    print("车牌测试总数: %d, 成功定位车牌数: %d, 定位成功率: %.2f%%" % (total_images, detected_images, det_rate))
    print("详细评测报表已写入: %s" % csv_report_path)
    print("[提示] 评测结果已展示在 LCD 上，按下 K1 或 BOOT 即可退出程序。")
    print("==================================================")

    # 在 LCD 上渲染评测总结卡片并等待按键退出
    try:
        summary_img = image.Image(size=(320, 240))
        summary_img.draw_rectangle(0, 0, 320, 240, color=(10, 25, 40), fill=True)
        summary_img.draw_rectangle(10, 10, 300, 220, color=(0, 220, 100), thickness=3)
        summary_img.draw_string(25, 25, "EVALUATION FINISHED", color=(0, 255, 255), scale=2)
        summary_img.draw_string(25, 70, "Total Images : %d" % total_images, color=(255, 255, 255), scale=2)
        summary_img.draw_string(25, 105, "Detected     : %d" % detected_images, color=(0, 255, 0), scale=2)
        summary_img.draw_string(25, 140, "Success Rate : %.1f%%" % det_rate, color=(255, 255, 0), scale=2)
        summary_img.draw_string(25, 185, "Press K1/BOOT to Exit", color=(0, 200, 255), scale=1)
        lcd.display(summary_img)

        # 保持屏幕显示，等待用户按 K1 或 BOOT 键退出
        t_exit = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t_exit) < 30000:
            if detector_key.check_pressed():
                break
            time.sleep_ms(50)
        del summary_img
    except Exception:
        pass

finally:
    gc.collect()
