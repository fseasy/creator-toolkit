
## Effects Comparison

SCRFD ~ Yolo8n > Mediapipe

模型都用的 nano 等最小的 model. SCRFD 可能误召回，但漏召情况好些 ；Mediapipe 肯定是不行的，直接删掉了。

Yolo model download: https://github.com/lindevs/yolov8-face?tab=readme-ov-file

SCRFD: https://github.com/deepinsight/insightface/tree/master/python-package (自己会下载)


目前脚本没有合并，两个模型各自在脚本里，用重复代码。以后再整理吧—直接codex 弄下就行，应该比较简单，有需要再弄。