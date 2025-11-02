from typing import Dict, List, Tuple
import numpy as np
import cv2

from orbit.nsvs.vlm.obj import DetectedObject


class VideoFrame:
    """Frame class."""
    def __init__(
        self,
        frame_idx: int,
        frame_images: Dict[str, List[np.ndarray]],
        object_of_interest: Dict[str, Tuple[str, DetectedObject]]
    ):
        self.frame_idx = frame_idx
        self.frame_images = frame_images
        self.object_of_interest = object_of_interest

    def save_frame_img(self, save_path: str) -> None:
        """Save frame image."""
        if self.frame_images is not None:
            for cam_id, images in self.frame_images.items():
                for idx, img in enumerate(images):
                    cv2.imwrite(f"{save_path}_{cam_id}_{idx}.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def thresholded_detected_objects(self, threshold) -> dict:
        """Get all detected object."""
        detected_obj = {}
        for prop, (cam_id, detected_object) in self.object_of_interest.items():
            probability = detected_object.get_detected_probability()
            if probability > threshold:
                detected_obj[prop] = (probability, cam_id)
        return detected_obj



