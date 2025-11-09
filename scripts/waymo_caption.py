import argparse
import base64
import json
import os
from typing import Any, Dict, List, Optional

import cv2
from nuscenes.nuscenes import NuScenes
from openai import OpenAI
from tqdm import tqdm


class VLLMCaptioner:
    """A client to generate captions using a vLLM server."""

    def __init__(self, api_base, model):
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self.model = model

    def _encode_image(self, frame):
        """Encodes a cv2 frame to a base64 string."""
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_caption(self, image, prompt):
        """Generates a caption for a single image using the vLLM server."""
        encoded_image = self._encode_image(image)

        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            },
        ]

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        return chat_response.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions for nuScenes scenes using vLLM, with object context from JSON."
    )
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataroot (e.g., /data/sets/nuscenes)")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version (e.g., v1.0-trainval, v1.0-mini)")
    parser.add_argument(
        "--scenes",
        required=True,
        help="Comma-separated scene names (e.g., scene-0061,scene-0103) or path to file with scene names, one per line",
    )
    parser.add_argument(
        "--json", required=True, help="Path to JSON file containing scene->camera->timestamp->[objects] structure"
    )
    parser.add_argument(
        "--cameras",
        default="",
        help="Comma-separated camera names to process (e.g., CAM_FRONT,CAM_FRONT_LEFT). If empty, process all cameras.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory for caption JSON files")
    parser.add_argument("--vllm-api", default="http://localhost:8001/v1", help="vLLM API base URL")
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="Model name for vLLM")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum number of frames to process per scene (for testing)"
    )
    return parser.parse_args()


def load_scene_names(scenes_arg: str) -> List[str]:
    """Load scene names from CLI arg (comma-separated) or file."""
    if os.path.isfile(scenes_arg):
        with open(scenes_arg, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = [s.strip() for s in scenes_arg.split(",") if s.strip()]
    return names


def load_json_data(json_path: str) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load aggregated JSON structure: {scene: {camera: {timestamp: [objects]}}}"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_objects_for_frame(
    json_data: Dict[str, Dict[str, Dict[str, List[str]]]], scene_id: str, camera: str, timestamp: str
) -> List[str]:
    """Look up object list for a given scene/camera/timestamp."""
    try:
        return json_data.get(scene_id, {}).get(camera, {}).get(str(timestamp), [])
    except (KeyError, AttributeError):
        return []


def iter_samples_for_scene(nusc: NuScenes, scene_record: Dict[str, Any]):
    """Iterate through all samples for a scene."""
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


def build_prompt(scene_id: str, camera: str, timestamp: str, objects: List[str]) -> str:
    """Build a prompt that includes object context."""
    if objects:
        objects_str = ", ".join(objects[:20])  # Limit to first 20 objects to avoid token limits
        if len(objects) > 20:
            objects_str += f" (and {len(objects) - 20} more)"
        prompt = (
            f"Objects detected in this scene (camera {camera}, scene {scene_id}, timestamp {timestamp}): "
            f"{objects_str}. "
            f"Based on both the image and this object list, describe the scene in one clear sentence."
        )
    else:
        prompt = (
            f"Describe this scene (camera {camera}, scene {scene_id}, timestamp {timestamp}) "
            f"in one clear sentence based on the image."
        )
    return prompt


def main():
    """Main function to run the captioning process."""
    args = parse_args()

    # Load configuration
    captioner = VLLMCaptioner(api_base=args.vllm_api, model=args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scene names and JSON data
    scene_names = load_scene_names(args.scenes)
    json_data = load_json_data(args.json)

    # Determine which cameras to process
    if args.cameras:
        cameras_to_process = [c.strip() for c in args.cameras.split(",") if c.strip()]
    else:
        cameras_to_process = CAMS

    # Initialize nuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Map scene names to scene records
    name_to_scene = {s["name"]: s for s in nusc.scene}
    scenes = [name_to_scene[name] for name in scene_names if name in name_to_scene]

    if not scenes:
        print(f"No valid scenes found from: {scene_names}")
        return

    # Process each scene
    all_output_data = {}
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_id = scene["name"]
        scene_data = {}

        # Process each camera
        for camera in cameras_to_process:
            camera_data = {}
            frame_count = 0

            # Iterate through samples for this scene
            for sample in iter_samples_for_scene(nusc, scene):
                if args.max_frames and frame_count >= args.max_frames:
                    break

                # Check if this camera has data for this sample
                if camera not in sample.get("data", {}):
                    continue

                timestamp = str(sample["timestamp"])
                sd_token = sample["data"][camera]

                # Get objects from JSON
                objects = get_objects_for_frame(json_data, scene_id, camera, timestamp)

                # Load image from nuScenes
                try:
                    # get_sample_data returns (data_path, boxes, camera_intrinsic)
                    data_path, _, _ = nusc.get_sample_data(sd_token)

                    if not os.path.exists(data_path):
                        print(f"Warning: Image not found: {data_path}")
                        continue

                    # Read image
                    img = cv2.imread(data_path)
                    if img is None:
                        print(f"Warning: Could not read image: {data_path}")
                        continue

                    # Build prompt with object context
                    prompt = build_prompt(scene_id, camera, timestamp, objects)

                    # Get caption
                    try:
                        caption = captioner.get_caption(img, prompt)
                        camera_data[timestamp] = caption
                        frame_count += 1
                    except Exception as e:
                        print(f"Error generating caption for {scene_id}/{camera}/{timestamp}: {e}")
                        continue

                except Exception as e:
                    print(f"Error processing {scene_id}/{camera}/{timestamp}: {e}")
                    continue

            if camera_data:
                scene_data[camera] = camera_data

        if scene_data:
            all_output_data[scene_id] = scene_data

            # Save per-scene JSON
            scene_output_path = os.path.join(args.output_dir, f"captions_{scene_id}.json")
            with open(scene_output_path, "w", encoding="utf-8") as f:
                json.dump({scene_id: scene_data}, f, indent=4)

    # Save combined output
    if all_output_data:
        combined_path = os.path.join(args.output_dir, "caption_output.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_output_data, f, indent=4)
        print(f"Finished captioning. Captions saved to {combined_path}")
    else:
        print("No captions generated.")


if __name__ == "__main__":
    main()
