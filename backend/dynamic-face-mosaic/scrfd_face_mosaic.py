import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

# Supported extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
VIDEO_FALLBACK_GRACE_FRAMES = 2
VIDEO_SMOOTHING_ALPHA = 0.65
MAX_TRACK_MATCH_DISTANCE_RATIO = 0.35
MIN_TRACK_HITS_FOR_PERSISTENCE = 2
MAX_PERSISTED_BOX_AREA_RATIO = 0.35
MISSED_FRAME_SHRINK_RATIO = 0.92
MIN_TRACK_MATCH_IOU = 0.05


class FaceObfuscator:
  """Detects faces with InsightFace SCRFD and applies a feathered blur."""

  def __init__(
    self,
    model_pack: str,
    model_root: Path,
    padding: float = 0.3,
    det_size: int = 640,
    det_thresh: float = 0.35,
  ):
    try:
      import onnxruntime as ort
      from insightface.app import FaceAnalysis
    except ImportError as exc:
      raise RuntimeError(
        "Missing dependencies. Install them with: uv pip install --python .venv/bin/python onnxruntime insightface"
      ) from exc

    providers = self._select_providers(ort.get_available_providers())
    self.app = FaceAnalysis(
      name=model_pack,
      root=str(model_root.expanduser()),
      allowed_modules=["detection"],
      providers=providers,
    )
    self.app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=(det_size, det_size))
    self.padding = padding
    self._video_tracks = []

  def _select_providers(self, available: list[str]) -> list[str]:
    preferred = ["GPUExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    selected = [provider for provider in preferred if provider in available]
    return selected or ["CPUExecutionProvider"]

  def process_image(self, frame: np.ndarray) -> np.ndarray:
    boxes = self._detect_boxes(frame)
    return self._apply_blur_boxes(frame, boxes)

  def process_video_frame(self, frame: np.ndarray) -> np.ndarray:
    boxes = self._detect_boxes(frame)
    stabilized_boxes = self._stabilize_video_boxes(frame.shape, boxes)
    return self._apply_blur_boxes(frame, stabilized_boxes)

  def _detect_boxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    faces = self.app.get(frame)
    boxes = []
    for face in faces:
      x1, y1, x2, y2 = face.bbox.astype(int).tolist()
      boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

  def _apply_blur_boxes(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = frame.shape[:2]
    if not boxes:
      return frame

    output_frame = frame.copy()
    for x, y, bw, bh in boxes:
      if bw <= 0 or bh <= 0:
        continue
      px = int(bw * self.padding)
      py = int(bh * self.padding)
      x1 = max(0, x - px)
      y1 = max(0, y - py)
      x2 = min(w, x + bw + px)
      y2 = min(h, y + bh + py)
      output_frame = self._apply_soft_blur(output_frame, x1, y1, x2, y2)
    return output_frame

  def _stabilize_video_boxes(
    self,
    frame_shape: tuple[int, ...],
    new_boxes: list[tuple[int, int, int, int]],
  ) -> list[tuple[int, int, int, int]]:
    active_tracks = []
    used_track_indexes = set()

    for box in new_boxes:
      best_match = self._find_best_track_match(box, used_track_indexes)
      if best_match is None:
        active_tracks.append({"box": box, "missed": 0, "hits": 1})
        continue

      matched_track = self._video_tracks[best_match]
      smoothed_box = self._smooth_box(matched_track["box"], box)
      active_tracks.append({"box": smoothed_box, "missed": 0, "hits": matched_track["hits"] + 1})
      used_track_indexes.add(best_match)

    for idx, track in enumerate(self._video_tracks):
      if idx in used_track_indexes:
        continue
      missed = track["missed"] + 1
      if missed > VIDEO_FALLBACK_GRACE_FRAMES:
        continue
      if track["hits"] < MIN_TRACK_HITS_FOR_PERSISTENCE:
        continue
      if self._box_area_ratio(track["box"], frame_shape[1], frame_shape[0]) > MAX_PERSISTED_BOX_AREA_RATIO:
        continue
      active_tracks.append(
        {
          "box": self._shrink_box(track["box"], MISSED_FRAME_SHRINK_RATIO),
          "missed": missed,
          "hits": track["hits"],
        }
      )

    self._video_tracks = active_tracks
    return [
      self._clip_box(track["box"], frame_shape[1], frame_shape[0])
      for track in self._video_tracks
      if track["box"][2] > 0 and track["box"][3] > 0
    ]

  def _find_best_track_match(self, box: tuple[int, int, int, int], used_indexes: set[int]) -> int | None:
    best_index = None
    best_distance = float("inf")
    new_cx, new_cy = self._box_center(box)
    new_w, new_h = box[2], box[3]
    distance_limit = max(new_w, new_h) * MAX_TRACK_MATCH_DISTANCE_RATIO

    for idx, track in enumerate(self._video_tracks):
      if idx in used_indexes:
        continue
      prev_box = track["box"]
      prev_cx, prev_cy = self._box_center(prev_box)
      center_distance = ((new_cx - prev_cx) ** 2 + (new_cy - prev_cy) ** 2) ** 0.5
      iou = self._box_iou(box, prev_box)
      size_ratio = max(new_w, 1) / max(prev_box[2], 1)
      if iou < MIN_TRACK_MATCH_IOU and center_distance > max(
        distance_limit, max(prev_box[2], prev_box[3]) * MAX_TRACK_MATCH_DISTANCE_RATIO
      ):
        continue
      if center_distance > max(distance_limit, max(prev_box[2], prev_box[3]) * MAX_TRACK_MATCH_DISTANCE_RATIO):
        continue
      if size_ratio < 0.5 or size_ratio > 2.0:
        continue
      if center_distance < best_distance:
        best_distance = center_distance
        best_index = idx

    return best_index

  def _smooth_box(
    self,
    previous_box: tuple[int, int, int, int],
    current_box: tuple[int, int, int, int],
  ) -> tuple[int, int, int, int]:
    return tuple(
      int(previous * VIDEO_SMOOTHING_ALPHA + current * (1 - VIDEO_SMOOTHING_ALPHA))
      for previous, current in zip(previous_box, current_box, strict=True)
    )  # type: ignore

  def _clip_box(self, box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    bw = max(1, min(bw, frame_w - x))
    bh = max(1, min(bh, frame_h - y))
    return x, y, bw, bh

  def _box_center(self, box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, bw, bh = box
    return x + bw / 2, y + bh / 2

  def _box_area_ratio(self, box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> float:
    return (box[2] * box[3]) / max(frame_w * frame_h, 1)

  def _shrink_box(self, box: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    new_w = max(1, int(bw * ratio))
    new_h = max(1, int(bh * ratio))
    dx = (bw - new_w) // 2
    dy = (bh - new_h) // 2
    return x + dx, y + dy, new_w, new_h

  def _box_iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
      return 0.0
    return inter_area / union

  def reset_video_state(self):
    self._video_tracks = []

  def _apply_soft_blur(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
      return image

    roi_h, roi_w = roi.shape[:2]
    blur_kernel = int(max(roi_w, roi_h) * 0.4) | 1
    blurred_roi = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)

    mask = np.zeros_like(roi, dtype=np.float32)
    center = (roi_w // 2, roi_h // 2)
    axes = (roi_w // 2, roi_h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (1, 1, 1), -1)

    mask_blur_kernel = int(max(roi_w, roi_h) * 0.2) | 1
    mask = cv2.GaussianBlur(mask, (mask_blur_kernel, mask_blur_kernel), 0)

    blended_roi = roi * (1 - mask) + blurred_roi * mask
    image[y1:y2, x1:x2] = blended_roi.astype(np.uint8)
    return image


class MediaProcessor:
  """Reads media, applies face blur, and preserves audio for videos."""

  def __init__(self, obfuscator: FaceObfuscator):
    self.obfuscator = obfuscator
    self.ffmpeg = shutil.which("ffmpeg")

  def process_image(self, in_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(in_path))
    if img is None:
      return False

    processed_img = self.obfuscator.process_image(img)
    cv2.imwrite(str(out_path), processed_img)
    return True

  def process_video(self, in_path: Path, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
      return False

    self.obfuscator.reset_video_state()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
      fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = out_path.with_suffix(".mp4")
    temp_video_path = self._temp_video_path(out_path)

    writer = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (w, h))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    try:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        processed = self.obfuscator.process_video_frame(frame)
        writer.write(processed)

        current_frame += 1
        if current_frame % 30 == 0:
          print(f"  -> Processing video: {current_frame}/{frame_count} frames", end="\r")
    finally:
      print(f"  -> Processing video: Done!{' ' * 20}")
      cap.release()
      writer.release()

    if not self._mux_original_audio(temp_video_path, in_path, out_path):
      if out_path.exists():
        out_path.unlink()
      temp_video_path.replace(out_path)
    elif temp_video_path.exists():
      temp_video_path.unlink()

    return True

  def _temp_video_path(self, out_path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(prefix=f"{out_path.stem}.", suffix=".video_only.mp4", dir=out_path.parent)
    os.close(fd)
    return Path(temp_name)

  def _mux_original_audio(self, temp_video_path: Path, source_video_path: Path, out_path: Path) -> bool:
    if self.ffmpeg is None:
      print("  [!] ffmpeg not found, output video will be saved without audio.")
      return False

    cmd = [
      self.ffmpeg,
      "-y",
      "-loglevel",
      "error",
      "-i",
      str(temp_video_path),
      "-i",
      str(source_video_path),
      "-map",
      "0:v:0",
      "-map",
      "1:a?",
      "-c:v",
      "copy",
      "-c:a",
      "copy",
      "-map_metadata",
      "1",
      "-movflags",
      "+faststart",
      "-shortest",
      str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
      return True

    print("  [!] Failed to mux original audio with ffmpeg:")
    stderr = result.stderr.strip()
    if stderr:
      print(f"      {stderr}")
    return False


def resolve_inputs(inputs: list[Path]) -> list[tuple[Path, Path]]:
  files_to_process = []
  for inp in inputs:
    inp = inp.resolve()
    if inp.is_file():
      files_to_process.append((inp, inp.parent))
    elif inp.is_dir():
      for filepath in inp.rglob("*"):
        if filepath.is_file():
          files_to_process.append((filepath, inp))
  return files_to_process


def process_paths(
  inputs: list[Path],
  output_dir: Path,
  model_pack: str,
  model_root: Path,
  padding: float,
  det_size: int,
  det_thresh: float,
):
  obfuscator = FaceObfuscator(
    model_pack=model_pack,
    model_root=model_root,
    padding=padding,
    det_size=det_size,
    det_thresh=det_thresh,
  )
  processor = MediaProcessor(obfuscator)
  files_to_process = resolve_inputs(inputs)
  output_dir = output_dir.resolve()

  main_pbar = tqdm(files_to_process, desc="Overall Progress", unit="file")
  for file_path, base_path in main_pbar:
    ext = file_path.suffix.lower()
    is_img = ext in IMG_EXTS
    is_vid = ext in VID_EXTS

    if not (is_img or is_vid):
      print(f"  [!] Skip non-target file: {file_path}")
      continue

    main_pbar.set_description(f"Processing: {file_path.name}")
    try:
      rel_path = file_path.relative_to(base_path)
    except ValueError:
      rel_path = Path(file_path.name)

    out_file_path = output_dir / rel_path
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing: {file_path.name}")

    success = (
      processor.process_image(file_path, out_file_path) if is_img else processor.process_video(file_path, out_file_path)
    )
    if not success:
      print(f"  [!] Failed to read or process: {file_path.name}")

  main_pbar.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Apply SCRFD-based face blur to images and videos while preserving video audio."
  )
  parser.add_argument("-i", "--input", nargs="+", type=Path, required=True, help="Input file(s) or folder(s)")
  parser.add_argument("-o", "--output", type=Path, required=True, help="Output folder")
  parser.add_argument(
    "--model-pack", default="buffalo_sc", help="InsightFace model pack name. buffalo_sc uses SCRFD-500MF."
  )
  parser.add_argument(
    "--model-root",
    type=Path,
    default=Path(__file__).resolve().parent / "model/insightface",
    help="Directory where InsightFace model packs are cached",
  )
  parser.add_argument("--padding", type=float, default=0.35, help="Extra padding ratio around each detected face")
  parser.add_argument("--det-size", type=int, default=640, help="Detection input size, e.g. 640 or 800")
  parser.add_argument("--det-thresh", type=float, default=0.35, help="Detection confidence threshold")

  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)

  process_paths(
    inputs=args.input,
    output_dir=args.output,
    model_pack=args.model_pack,
    model_root=args.model_root,
    padding=args.padding,
    det_size=args.det_size,
    det_thresh=args.det_thresh,
  )
  print("\nAll tasks completed successfully.")
