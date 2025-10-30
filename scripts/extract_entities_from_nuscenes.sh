#!/bin/bash

# python extract_entities_from_nuscenes.py \
#   --dataroot /nas/standard_datasets/nuscenes \
#   --version v1.0-trainval \
#   --scenes scene-0061 \
#   --output scene-0061.json \
#   --pretty \
#   --per-camera \
#   --skip-categories movable_object.barrier

python nuscenes_overlay_video.py \
  --dataroot /nas/standard_datasets/nuscenes \
  --version v1.0-trainval \
  --scene scene-0061 \
  --camera CAM_FRONT_LEFT \
  --input-json scene-0061.json \
  --output scene-0061_CAM_FRONT_LEFT.mp4 \
  --fps 2