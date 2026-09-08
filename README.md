# K210 视觉实验

面向 K210 / CanMV 开发板的课程实验代码，包含按键诊断、手部检测、车牌识别和手势数字识别。

> 仓库只包含实验代码与少量自采样例图。课程讲义、厂商模型、固件、烧录工具和 IDE 安装包保留在本地，因来源和再分发许可未确认而没有上传。

## 目录

```text
experiments/
  03-hand-detection/              手部采集、实时检测、批量统计
  04-license-plate-recognition/   车牌采集、实时/批量识别
  05-gesture-recognition/         手势数字识别
tools/
  key_diagnostic.py               K1 / BOOT 按键诊断
assets/sample-license-plates/     5 张自采车牌样例图
docs/
  setup.md                        环境和模型准备说明
```

## 运行环境

1. 使用与开发板匹配的 CanMV 固件和 CanMV IDE，将需要运行的单个 `.py` 文件上传至开发板。
2. 各程序面向 K210 的 MicroPython API（`sensor`、`lcd`、`image`、`maix` 等），不能直接在电脑的 CPython 中运行。
3. 准备模型文件并复制到 SD 卡。程序使用的默认路径如下：

   - 手部检测：`/sd/KPU/yolo_hand_detect/hand_detect.kmodel`
   - 车牌识别：`/sd/KPU/car_licenseplate_recog/` 下的检测模型、识别模型和权重；中文显示还需要 `/sd/0xA00000.Dzk`
   - 手势识别：`/sd/det.kmodel`（训练产物，未随仓库发布）

详细路径和使用方式见 [docs/setup.md](docs/setup.md)。

## 实验说明

| 模块 | 入口 | 功能 |
| --- | --- | --- |
| 按键诊断 | `tools/key_diagnostic.py` | K1（IO16）和 BOOT（默认 IO0）防抖、按键次数及 LCD 状态显示 |
| 实验三 | `experiments/03-hand-detection/` | 按键拍摄、YOLOv2 实时手部检测、批量推理统计 |
| 实验四 | `experiments/04-license-plate-recognition/` | 车牌拍摄、实时识别、测试集批量识别及高帧率版 |
| 实验五 | `experiments/05-gesture-recognition/` | 基于 YOLOv2 的 0–5 手势数字识别和多帧投票 |

## 注意事项

- 按键与 LCD 接线会随开发板版本变化；运行前核对 IO 映射。
- 车牌识别内存占用较高，通常需要与代码匹配的 Lite 固件。
- 车牌和手部图像可能含有可识别信息；提交新的数据前请确认已获得使用与公开授权。
