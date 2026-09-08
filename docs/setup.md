# 环境与资源准备

## 1. 烧录与连接

使用与所持 K210 开发板匹配的 CanMV 固件和 IDE。完成烧录后，通过串口连接开发板，并在 IDE 中打开本仓库中要运行的单个脚本后上传执行。

本仓库不包含任何固件、IDE、烧录工具或驱动。请从开发板厂商或 CanMV 官方渠道获取与硬件匹配、且允许使用的版本。

## 2. SD 卡目录

将模型复制至下列目录；目录名称须与程序中的路径一致。

```text
/sd/
├── KPU/
│   ├── yolo_hand_detect/
│   │   └── hand_detect.kmodel
│   └── car_licenseplate_recog/
│       ├── lp_detect.kmodel
│       ├── lp_recog.kmodel
│       └── lp_weight.bin
├── 0xA00000.Dzk            # 车牌中文字体
└── det.kmodel              # 实验五手势数字识别模型
```

模型、字体和固件为课程或厂商提供的二进制资源，本仓库未再分发。请按其原始许可和课程要求获取。

## 3. 运行入口

- `experiments/03-hand-detection/01_手部图像采集.py`：按 K1（IO16）或 BOOT（IO0）将图像存至 `/sd/hand_dataset`。
- `experiments/03-hand-detection/02_实时手部检测.py`：实时手部检测，可按键截取检测画面。
- `experiments/04-license-plate-recognition/02_实时车牌识别.py`：实时车牌检测与识别。
- `experiments/04-license-plate-recognition/04_实时车牌识别_高帧率版.py`：隔帧推理的高帧率版本。
- `experiments/05-gesture-recognition/gesture_recognition.py`：手势数字识别，要求 `/sd/det.kmodel`。
- `tools/key_diagnostic.py`：检查 K1 / BOOT 的防抖和触发状态。

## 4. 本机图片检查

`assets/sample-license-plates/` 只存放少量输入样例。它们用于本机查看或测试集脚本的输入参考，并非训练数据集。
