import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
from nuscenes.nuscenes import NuScenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MP4 for a nuScenes scene+camera with categories overlaid per keyframe.\n"
            "Input JSON must be in the shape: {scene_id: {CAM: {timestamp: [category_names]}}}."
        )
    )
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataroot (e.g., /data/sets/nuscenes)")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)")
    parser.add_argument("--scene", required=True, help="Scene id (e.g., scene-0061)")
    parser.add_argument(
        "--camera",
        required=True,
        choices=[
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
    )
    parser.add_argument("--input-json", required=True, help="Path to per-camera categories JSON produced earlier")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=2, help="Output video FPS (default 2, matches keyframes)")
    parser.add_argument("--max-lines", type=int, default=10, help="Max caption lines to draw")
    parser.add_argument("--font-scale", type=float, default=0.7, help="OpenCV font scale")
    parser.add_argument("--thickness", type=int, default=2, help="OpenCV text thickness")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="If set, de-duplicate category names per frame before displaying. Default: keep duplicates.",
    )
    return parser.parse_args()


def load_scene_camera_map(input_json: str, scene: str, camera: str) -> Dict[str, List[str]]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if scene not in data:
        raise ValueError(f"Scene {scene} not found in JSON")
    if camera not in data[scene]:
        raise ValueError(f"Camera {camera} not found under scene {scene} in JSON")
    ts_to_categories: Dict[str, List[str]] = data[scene][camera]
    return ts_to_categories


def iter_scene_camera_frames(nusc: NuScenes, scene_name: str, camera: str) -> List[Tuple[str, str]]:
    # Returns list of (timestamp_str, image_path) for this scene and camera.
    name_to_scene = {s["name"]: s for s in nusc.scene}
    if scene_name not in name_to_scene:
        raise ValueError(f"Scene {scene_name} not found in nuScenes metadata")
    scene_rec = name_to_scene[scene_name]

    frames: List[Tuple[str, str]] = []
    token = scene_rec["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        sd_token = sample["data"][camera]
        data_path, _, _ = nusc.get_sample_data(sd_token)
        frames.append((str(sample["timestamp"]), data_path))
        token = sample["next"]
    return frames


def draw_caption(
    image, header: str, categories: List[str], max_lines: int, font_scale: float, thickness: int, dedupe: bool
):
    # Create semi-transparent banner at the top
    overlay = image.copy()
    h, w = image.shape[:2]
    banner_h = min(int(h * 0.25), 200)
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    alpha = 0.5
    image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Prepare lines: header first, then categories (optionally deduped), wrapped by count
    if dedupe:
        cats_for_display = sorted(dict.fromkeys(categories).keys())
    else:
        cats_for_display = categories
    lines: List[str] = [header]
    # Simple wrapping by count per line: try to fit ~6-8 tokens per line depending on length
    current: List[str] = []
    max_tokens_per_line = 8
    for cat in cats_for_display:
        current.append(cat)
        if len(current) >= max_tokens_per_line:
            lines.append(", ".join(current))
            current = []
    if current:
        lines.append(", ".join(current))

    # Clamp to max_lines
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["…"]

    y = 30
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(image, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += int(30 * font_scale) + 8


def main() -> None:
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    ts_to_categories = load_scene_camera_map(args.input_json, args.scene, args.camera)
    frames = iter_scene_camera_frames(nusc, args.scene, args.camera)

    # Initialize writer based on first frame
    if not frames:
        raise RuntimeError("No frames found for the specified scene and camera")
    first_ts, first_path = frames[0]
    first_img = cv2.imread(first_path)
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {first_path}")
    height, width = first_img.shape[:2]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))

    try:
        for ts, img_path in frames:
            img = cv2.imread(img_path)
            if img is None:
                # Skip unreadable frames but keep timeline moving
                img = first_img.copy()
            cats = ts_to_categories.get(ts, [])
            header = f"{args.scene} | {args.camera} | {ts}"
            draw_caption(img, header, cats, args.max_lines, args.font_scale, args.thickness, args.dedupe)
            writer.write(img)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
