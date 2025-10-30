#!/bin/bash

# python extract_entities_from_nuscenes.py \
#   --dataroot /nas/standard_datasets/nuscenes \
#   --version v1.0-trainval \
#   --scenes scene-0061,scene-0103


# python extract_entities_from_nuscenes.py \
#   --dataroot /nas/standard_datasets/nuscenes \
#   --version v1.0-trainval \
#   --scenes scene-0061 \
#   --output scene-0061.json \
#   --pretty

python extract_entities_from_nuscenes.py \
  --dataroot /nas/standard_datasets/nuscenes \
  --version v1.0-trainval \
  --scenes scene-0061 \
  --output scene-0061.json \
  --pretty \
  --per-camera
