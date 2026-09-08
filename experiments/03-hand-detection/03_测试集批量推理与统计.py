# =====================================================================
# 实验三：YOLOv2 手部测试集科研级量化评估系统 (稳健版)
# =====================================================================
import lcd, image, time, os, gc
from maix import KPU

lcd.init()
lcd.clear(lcd.BLACK)

resize_img = image.Image(size=(320, 256))
anchor = (0.8125, 0.4556, 1.1328, 1.2667, 1.8594, 1.4889, 1.4844, 2.2000, 2.6484, 2.9333)
names = ['hand']

model_path = "/sd/KPU/yolo_hand_detect/hand_detect.kmodel"
try:
    with open(model_path, "rb") as f:
        pass
except Exception:
    model_path = "/sd/hand_detect.kmodel"

print(">>> 加载手部检测模型:", model_path)

hand_kpu = KPU()
hand_kpu.load_kmodel(model_path)
hand_kpu.init_yolo2(
    anchor,
    anchor_num=len(anchor) // 2,
    img_w=320, img_h=240,
    net_w=320, net_h=256,
    layer_w=10, layer_h=8,
    threshold=0.45,
    nms_value=0.3,
    classes=len(names)
)

dataset_dir = "/sd/hand_dataset"
csv_report_path = "/sd/hand_eval_report.csv"

try:
    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".jpg")]
    files.sort()
except Exception as e:
    print("读取数据集目录异常:", e)
    files = []

total_samples = len(files)
if total_samples == 0:
    print("[错误] 未在 %s 找到测试图片，请先运行 01_手部图像采集.py 拍照！" % dataset_dir)
    raise SystemExit

print("================================================================")
print("       YOLOv2 手部检测科研量化评测体系启动")
print("       样本总数: %d 张 | 报表保存至: %s" % (total_samples, csv_report_path))
print("================================================================")

try:
    with open(csv_report_path, "w") as csv:
        csv.write("Index,FileName,Status,Confidence,Box_X,Box_Y,Box_W,Box_H,Time_Pre_ms,Time_KPU_ms,Time_NMS_ms,Time_Total_ms\n")
except Exception as e:
    print("创建 CSV 失败:", e)

tp_count = 0
fn_count = 0
fp_count = 0
sum_kpu_time = 0
sum_total_time = 0

for i, fname in enumerate(files):
    gc.collect()
    fpath = "%s/%s" % (dataset_dir, fname)
    
    try:
        t_start = time.ticks_ms()
        img = image.Image(fpath)
        
        t_p0 = time.ticks_ms()
        resize_img.draw_image(img, 0, 0).pix_to_ai()
        t_pre = time.ticks_ms() - t_p0
        
        t_k0 = time.ticks_ms()
        hand_kpu.run_with_output(resize_img)
        t_kpu = time.ticks_ms() - t_k0
        
        t_n0 = time.ticks_ms()
        hands = hand_kpu.regionlayer_yolo2()
        t_nms = time.ticks_ms() - t_n0
        
        t_total = time.ticks_ms() - t_start
        sum_kpu_time += t_kpu
        sum_total_time += t_total
        
        if hands:
            tp_count += 1
            if len(hands) > 1:
                fp_count += (len(hands) - 1)
                
            best_hand = hands[0]
            conf = best_hand[5]
            bx, by, bw, bh = best_hand[0], best_hand[1], best_hand[2], best_hand[3]
            
            img.draw_rectangle(bx, by, bw, bh, color=(0, 255, 0), thickness=2)
            img.draw_string(bx + 2, by + 2, "%.2f" % conf, color=(0, 255, 0), scale=1)
            
            print("[%02d/%02d] %s -> 检出! Conf: %.2f | 框:(%d,%d,%d,%d) | KPU: %dms" % 
                  (i + 1, total_samples, fname, conf, bx, by, bw, bh, t_kpu))
            
            with open(csv_report_path, "a") as csv:
                csv.write("%d,%s,%s,%.4f,%d,%d,%d,%d,%d,%d,%d,%d\n" % 
                          (i + 1, fname, "DETECTED", conf, bx, by, bw, bh, t_pre, t_kpu, t_nms, t_total))
        else:
            fn_count += 1
            img.draw_string(20, 20, "MISSED (FN)", color=(255, 0, 0), scale=2)
            print("[%02d/%02d] %s -> 漏检 (False Negative) | 耗时: %dms" % 
                  (i + 1, total_samples, fname, t_total))
            
            with open(csv_report_path, "a") as csv:
                csv.write("%d,%s,%s,0,0,0,0,0,%d,%d,%d,%d\n" % 
                          (i + 1, fname, "MISSED", t_pre, t_kpu, t_nms, t_total))
        
        img.draw_rectangle(0, 0, 320, 18, color=(30, 30, 30), fill=True)
        img.draw_string(5, 2, "[%d/%d] %s" % (i + 1, total_samples, fname), color=(0, 255, 255), scale=1)
        lcd.display(img)
        del img
        time.sleep_ms(250)
        
    except Exception as e:
        print("[%02d/%02d] %s 推理失败:" % (i + 1, total_samples, fname), e)

# 指标计算
precision = (tp_count / (tp_count + fp_count)) * 100 if (tp_count + fp_count) > 0 else 0
recall = (tp_count / (tp_count + fn_count)) * 100 if (tp_count + fn_count) > 0 else 0
f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
avg_kpu = sum_kpu_time / total_samples if total_samples > 0 else 0
avg_total = sum_total_time / total_samples if total_samples > 0 else 0
detection_rate = (tp_count / total_samples) * 100 if total_samples > 0 else 0

print("\n" + "=" * 60)
print("       【YOLOv2 手部检测科研量化评测最终报告】")
print("=" * 60)
print("1. 样本总体规模 (Total Samples)   : %d 张" % total_samples)
print("2. 混淆统计矩阵 (Confusion Matrix):")
print("   - 正确检出数 (True Positive, TP)  : %d" % tp_count)
print("   - 漏检目标数 (False Negative, FN): %d" % fn_count)
print("   - 误报虚警数 (False Positive, FP): %d" % fp_count)
print("3. 目标检测核心指标 (Core Metrics):")
print("   - 精确率 (Precision)              : %.2f%%" % precision)
print("   - 召回率 (Recall)                 : %.2f%%" % recall)
print("   - 综合平衡指数 (F1-Score)         : %.2f" % f1_score)
print("   - 样本检出率 (Detection Rate)     : %.2f%%" % detection_rate)
print("4. 算力性能剖析 (Compute Profiling):")
print("   - 平均单帧 KPU 纯推理耗时         : %.1f ms" % avg_kpu)
print("   - 平均单帧全流程总时延           : %.1f ms (约 %.1f FPS)" % (avg_total, 1000.0 / avg_total if avg_total > 0 else 0))
print("5. 结构化报表已保存至: %s" % csv_report_path)
print("=" * 60 + "\n")

rep_img = image.Image(size=(320, 240))
rep_img.clear()
rep_img.draw_rectangle(5, 5, 310, 230, color=(0, 200, 200), thickness=2)
rep_img.draw_string(20, 15, "EVALUATION REPORT", color=(0, 255, 255), scale=2)
rep_img.draw_string(20, 50, "Samples  : %d" % total_samples, color=(255, 255, 255), scale=1)
rep_img.draw_string(170, 50, "DetRate: %.1f%%" % detection_rate, color=(0, 255, 0), scale=1)
rep_img.draw_string(20, 75, "Precision: %.1f%%" % precision, color=(255, 255, 0), scale=1)
rep_img.draw_string(170, 75, "Recall : %.1f%%" % recall, color=(255, 255, 0), scale=1)
rep_img.draw_string(20, 100, "F1-Score : %.2f" % f1_score, color=(255, 150, 0), scale=2)
rep_img.draw_string(20, 135, "KPU Time : %.1f ms" % avg_kpu, color=(200, 200, 200), scale=1)
rep_img.draw_string(170, 135, "Latency: %.1f ms" % avg_total, color=(200, 200, 200), scale=1)
rep_img.draw_string(20, 165, "TP=%d, FN=%d, FP=%d" % (tp_count, fn_count, fp_count), color=(180, 180, 255), scale=1)
rep_img.draw_string(20, 195, "Saved -> /sd/hand_eval_report.csv", color=(100, 255, 100), scale=1)
lcd.display(rep_img)

hand_kpu.deinit()
del hand_kpu
gc.collect()
