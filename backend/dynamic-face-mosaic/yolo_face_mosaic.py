import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Suppress YOLO logging
os.environ["YOLO_VERBOSE"] = "False"


def get_device():
  try:
    import torch

    if torch.backends.mps.is_available():
      return "mps"
  except Exception:
    pass
  return "cpu"


class FaceObfuscator:
  """Uses YOLOv8-face with Object Tracking for maximum stability on Mac."""

  def __init__(self, padding: float = 0.3):
    model_path = Path(__file__).resolve().parent / "model/yolov8n-face-lindevs.pt"
    self.model = YOLO(model_path)
    self.padding = padding
    self.device = get_device()

  def process_frame(self, frame: np.ndarray, is_video: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]

    if is_video:
      results = self.model.track(
        frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml", verbose=False, device=self.device
      )
    else:
      results = self.model.predict(frame, conf=0.3, verbose=False, device=self.device)

    if not results or len(results[0].boxes) == 0:
      return frame

    output_frame = frame.copy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
      x1, y1, x2, y2 = box.astype(int)
      bw, bh = x2 - x1, y2 - y1

      px, py = int(bw * self.padding), int(bh * self.padding)
      nx1, ny1 = max(0, x1 - px), max(0, y1 - py)
      nx2, ny2 = min(w, x2 + px), min(h, y2 + py)

      output_frame = self._apply_soft_blur(output_frame, nx1, ny1, nx2, ny2)

    return output_frame

  def _apply_soft_blur(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
      return image

    rh, rw = roi.shape[:2]
    blur_k = int(max(rw, rh) * 0.5) | 1
    blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

    mask = np.zeros((rh, rw, 3), dtype=np.float32)
    cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, (1, 1, 1), -1)

    feather_k = int(max(rw, rh) * 0.2) | 1
    mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)

    blended = roi.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    image[y1:y2, x1:x2] = blended.astype(np.uint8)
    return image


class MediaProcessor:
  def __init__(self, obfuscator: FaceObfuscator):
    self.obfuscator = obfuscator

  def process_image(self, in_p: Path, out_p: Path):
    img = cv2.imread(str(in_p))
    if img is not None:
      processed = self.obfuscator.process_frame(img, is_video=False)
      cv2.imwrite(str(out_p), processed)

  def process_video(self, in_p: Path, out_p: Path):
    cap = cv2.VideoCapture(str(in_p))
    if not cap.isOpened():
      return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # a video without audio
    temp_out = out_p.with_name(f"temp_{out_p.name}").with_suffix(".mp4")
    writer = cv2.VideoWriter(str(temp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with tqdm(total=total, desc=f"  [Video] {in_p.name[:20]}", leave=False) as pbar:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        processed = self.obfuscator.process_frame(frame, is_video=True)
        writer.write(processed)
        pbar.update(1)

    cap.release()
    writer.release()

    # 2. mux audio
    final_out = out_p.with_suffix(".mp4")
    self._merge_audio(in_p, temp_out, final_out)

    if temp_out.exists():
      temp_out.unlink()

  def _merge_audio(self, original_vid: Path, processed_vid: Path, output_vid: Path):
    """利用 FFmpeg 将原视频的音频无损提取并合并到处理后的视频中"""
    # 组装 ffmpeg 命令
    # -map 0:v:0 取 processed_vid 的视频流
    # -map 1:a:0? 取 original_vid 的音频流 (?表示如果原视频没声音也不报错)
    # -c:v copy 视频流直接复制，无需重新编码（极快）
    cmd = [
      "ffmpeg",
      "-y",
      "-i",
      str(processed_vid),
      "-i",
      str(original_vid),
      "-c:v",
      "copy",
      "-c:a",
      "aac",
      "-map",
      "0:v:0",
      "-map",
      "1:a:0?",
      "-shortest",
      str(output_vid),
    ]

    try:
      # 运行 FFmpeg (隐藏烦人的输出信息)
      subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
      print(f"\n[警告] 音频合并失败: {original_vid.name}。视频将没有声音。")
      shutil.copy(str(processed_vid), str(output_vid))
    except FileNotFoundError:
      print("\n[警告] 找不到 FFmpeg 插件！视频已保存但没有声音。请在终端执行 'brew install ffmpeg'。")
      shutil.copy(str(processed_vid), str(output_vid))


def main():
  parser = argparse.ArgumentParser(description="AI Face Soft Blur with YOLO Tracking")
  parser.add_argument("-i", "--input", nargs="+", type=Path, required=True)
  parser.add_argument("-o", "--output", type=Path, required=True)
  args = parser.parse_args()

  obf = FaceObfuscator(padding=0.3)
  proc = MediaProcessor(obf)

  tasks = []
  img_exts = {".jpg", ".jpeg", ".png", ".webp"}
  vid_exts = {".mp4", ".mov", ".avi", ".mkv"}

  for inp in args.input:
    root = inp if inp.is_dir() else inp.parent
    files = inp.rglob("*") if inp.is_dir() else [inp]
    for f in files:
      if f.is_file() and f.suffix.lower() in (img_exts | vid_exts):
        tasks.append((f, root))

  with tqdm(total=len(tasks), desc="Overall Progress") as main_pbar:
    for f_path, base in tasks:
      main_pbar.set_description(f"Processing: {f_path.name}")

      try:
        rel = f_path.relative_to(base)
      except ValueError:
        rel = Path(f_path.name)

      out_f = args.output / rel
      out_f.parent.mkdir(parents=True, exist_ok=True)

      if f_path.suffix.lower() in vid_exts:
        proc.process_video(f_path, out_f)
      else:
        proc.process_image(f_path, out_f)

      main_pbar.update(1)


if __name__ == "__main__":
  main()
