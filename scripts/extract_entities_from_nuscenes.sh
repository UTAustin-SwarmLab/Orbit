#!/bin/bash

# python extract_entities_from_nuscenes.py \
#   --dataroot /nas/standard_datasets/nuscenes \
#   --version v1.0-trainval \
#   --scenes scene-0061 \
#   --output scene-0061.json \
#   --pretty \
#   --per-camera \
#   --skip-categories movable_object.barrier \
#   --with-attributes

# python nuscenes_overlay_video.py \
#   --dataroot /nas/standard_datasets/nuscenes \
#   --version v1.0-trainval \
#   --scene scene-0061 \
#   --camera CAM_FRONT_LEFT \
#   --input-json scene-0061.json \
#   --output scene-0061_CAM_FRONT_LEFT.mp4 \
#   --fps 2

python waymo_caption.py \
  --dataroot /nas/standard_datasets/nuscenes \
  --version v1.0-trainval \
  --scenes scene-0061 \
  --json scripts/scene-0061.json \
  --output-dir outputs \
  --vllm-api http://localhost:8001/v1 \
  --model llava-hf/llava-1.5-7b-hf
