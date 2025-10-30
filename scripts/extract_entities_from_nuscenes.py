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
    parser.add_argument(
        "--skip-categories",
        default="",
        help=(
            "Comma-separated category names to exclude from output (e.g., movable_object.barrier,vehicle.trafficcone)."
        ),
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


def per_camera_objects(nusc: "NuScenes", sample: Dict[str, Any], skip: set) -> Dict[str, List[str]]:
    cam_to_sd = {c: sample["data"][c] for c in CAMS if c in sample["data"]}
    ann_tokens = set(sample["anns"])
    per_cam: Dict[str, List[str]] = {}
    for cam, sd_tk in cam_to_sd.items():
        _, boxes, _ = nusc.get_sample_data(
            sd_tk,
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=list(ann_tokens),
        )
        cats = [nusc.get("sample_annotation", b.token)["category_name"] for b in boxes]
        per_cam[cam] = [c for c in cats if c not in skip]
    return per_cam


def extract_per_frame_objects(
    dataroot: str, version: str, scene_names: List[str], per_camera: bool = False, skip: set | None = None
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

    skip = skip or set()

    for scene in scenes:
        for sample in iter_samples_for_scene(nusc, scene):
            if per_camera:
                per_cam = per_camera_objects(nusc, sample, skip)
                # Nest as: scene -> camera -> timestamp -> [category_names]
                record = {
                    "scene": scene["name"],
                    "camera": {cam: {str(sample["timestamp"]): cats} for cam, cats in per_cam.items()},
                }
            else:
                # Minimal per-sample list of category names (no camera breakdown)
                category_names = [
                    nusc.get("sample_annotation", ann_token)["category_name"] for ann_token in sample["anns"]
                ]
                category_names = [c for c in category_names if c not in skip]
                record = {
                    "scene": scene["name"],
                    "timestamp": sample["timestamp"],
                    "object_category_names": category_names,
                }

            yield record


def main() -> None:
    args = parse_args()
    scene_names = load_scene_names(args.scenes, args.scenes_file)
    skip = set(s.strip() for s in args.skip_categories.split(",") if s.strip())
    if args.pretty:
        if args.per_camera:
            # Aggregate into {scene_id: {CAM: {timestamp: [categories]}}}
            aggregated: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
            for record in extract_per_frame_objects(
                args.dataroot, args.version, scene_names, per_camera=True, skip=skip
            ):
                scene_id = record["scene"]
                cam_map = record["camera"]
                if scene_id not in aggregated:
                    aggregated[scene_id] = {}
                for cam, ts_map in cam_map.items():
                    if cam not in aggregated[scene_id]:
                        aggregated[scene_id][cam] = {}
                    for ts, cats in ts_map.items():
                        aggregated[scene_id][cam][ts] = cats
            data_obj = aggregated
        else:
            data_obj = list(
                extract_per_frame_objects(args.dataroot, args.version, scene_names, per_camera=False, skip=skip)
            )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data_obj, f, indent=2)
                f.write("\n")
        else:
            json.dump(data_obj, sys.stdout, indent=2)
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        handle = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        try:
            for record in extract_per_frame_objects(
                args.dataroot, args.version, scene_names, per_camera=args.per_camera, skip=skip
            ):
                handle.write(json.dumps(record) + "\n")
            handle.flush()
        finally:
            if handle is not sys.stdout:
                handle.close()


if __name__ == "__main__":
    main()
