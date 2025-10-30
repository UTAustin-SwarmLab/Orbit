import argparse
import json
import sys
from typing import Any, Dict, Iterable, List

from nuscenes.nuscenes import NuScenes

try:
    from nuscenes.utils.data_classes import BoxVisibility  # type: ignore
except Exception:  # pragma: no cover
    # Fallback for older nuscenes-devkit where BoxVisibility may be absent.
    class _BoxVisibilityFallback:
        ANY = 1

    BoxVisibility = _BoxVisibilityFallback()  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-keyframe object lists from nuScenes scenes.\n"
            "- Default: JSONL to stdout (one object per line).\n"
            "- Use --pretty for a single pretty JSON array.\n"
            "- Use --per-camera to add per-camera lists per sample."
        ),
    )
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataroot (e.g., /data/sets/nuscenes)")
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)",
    )
    parser.add_argument(
        "--scenes",
        default="",
        help="Comma-separated scene names to include (e.g., scene-0061,scene-0103). If empty, process all scenes.",
    )
    parser.add_argument(
        "--scenes-file",
        default="",
        help="Optional path to a file containing scene names, one per line. Combined with --scenes.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save output. If omitted, writes to stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (writes a single JSON array). Default without this flag is JSONL (one object per line).",
    )
    parser.add_argument(
        "--per-camera",
        action="store_true",
        help="Include per-camera object lists for each sample (uses camera frustum and image bounds).",
    )
    return parser.parse_args()


def load_scene_names(cli_scenes: str, scenes_file: str) -> List[str]:
    names: List[str] = []
    if cli_scenes.strip():
        names.extend([s.strip() for s in cli_scenes.split(",") if s.strip()])
    if scenes_file.strip():
        with open(scenes_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def iter_samples_for_scene(nusc, scene_record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    token = scene_record["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


# nuScenes camera channels
CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def per_camera_objects(nusc: "NuScenes", sample: Dict[str, Any], include_fields_fn) -> Dict[str, List[Dict[str, Any]]]:
    cam_to_sd = {c: sample["data"][c] for c in CAMS if c in sample["data"]}
    ann_tokens = set(sample["anns"])
    per_cam: Dict[str, List[Dict[str, Any]]] = {}
    for cam, sd_tk in cam_to_sd.items():
        _, boxes, _ = nusc.get_sample_data(
            sd_tk,
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=list(ann_tokens),
        )
        per_cam[cam] = [include_fields_fn(nusc.get("sample_annotation", b.token)) for b in boxes]
    return per_cam


def extract_per_frame_objects(
    dataroot: str, version: str, scene_names: List[str], per_camera: bool = False
) -> Iterable[Dict[str, Any]]:
    if NuScenes is None:
        raise RuntimeError("nuscenes-devkit is not installed. Install with: pip install nuscenes-devkit")

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # Select scenes
    if scene_names:
        name_to_scene = {s["name"]: s for s in nusc.scene}
        scenes = [name_to_scene[name] for name in scene_names if name in name_to_scene]
    else:
        scenes = list(nusc.scene)

    for scene in scenes:
        for sample in iter_samples_for_scene(nusc, scene):
            objects = []

            def include_ann_fields(ann: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "token": ann["token"],
                    "instance_token": ann["instance_token"],
                    "category_name": ann["category_name"],
                    "visibility_token": ann["visibility_token"],
                    "size": ann["size"],
                    "translation": ann["translation"],
                    "rotation": ann["rotation"],
                    "attribute_tokens": ann["attribute_tokens"],
                }

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                objects.append(include_ann_fields(ann))

            record = {
                "scene": scene["name"],
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "objects": objects,
            }

            if per_camera:
                record["per_camera"] = per_camera_objects(nusc, sample, include_ann_fields)

            yield record


def main() -> None:
    args = parse_args()
    scene_names = load_scene_names(args.scenes, args.scenes_file)
    if args.pretty:
        data = list(extract_per_frame_objects(args.dataroot, args.version, scene_names, per_camera=args.per_camera))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        else:
            json.dump(data, sys.stdout, indent=2)
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        handle = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            for record in extract_per_frame_objects(
                args.dataroot, args.version, scene_names, per_camera=args.per_camera
            ):
                handle.write(json.dumps(record) + "\n")
            handle.flush()
        finally:
            if handle is not sys.stdout:
                handle.close()


if __name__ == "__main__":
    main()
