
## Effects Comparison

SCRFD ~ Yolo8n > Mediapipe

模型都用的 nano 等最小的 model. SCRFD 可能误召回，但漏召情况好些 ；Mediapipe 肯定是不行的，直接删掉了。

yolo8n 对长序列速度有点慢。感觉 yolo8n 效果更稳定点，特别是对单帧图片？不知道是偶发还是bug 还是阈值的问题，反正后面处理图片的时候，SCRFD 完全失效了。

Yolo model download: https://github.com/lindevs/yolov8-face?tab=readme-ov-file

SCRFD: https://github.com/deepinsight/insightface/tree/master/python-package (自己会下载)


