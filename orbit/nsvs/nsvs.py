import numpy as np
import warnings
import bisect
import tqdm
import os

from orbit.nsvs.model_checker.property_checker import PropertyChecker
from orbit.nsvs.model_checker.video_automaton import VideoAutomaton
from orbit.nsvs.video.frames_of_interest import FramesofInterest
from orbit.utils.intersection import intersection_with_gaps
from orbit.nsvs.video.video_frame import VideoFrame
from orbit.nsvs.vlm.vllm_client import VLLMClient


PRINT_ALL = False
warnings.filterwarnings("ignore")

from orbit.nsvs.vlm.obj import DetectedObject

def run_nsvs(
    multi_video_data: list,
    video_paths: list,
    proposition: list,
    specification: str,
    model_name: str,
    device: int,
    model_type: str = "dtmc",
    num_of_frame_in_sequence = 3,
    tl_satisfaction_threshold: float = 0.6,
    detection_threshold: float = 0.5,
    vlm_detection_threshold: float = 0.349,
    image_output_dir: str = "outputs"
):
    """Find relevant frames from a video that satisfy a specification"""

    if PRINT_ALL:
        print(f"\nPropositions: {proposition}")
        print(f"Specification: {specification}")
        print(f"Video path: {video_paths}\n")

    vlm = VLLMClient(model=model_name, api_base=f"http://localhost:800{device}/v1")

    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)

    checker = PropertyChecker(
        proposition=proposition,
        specification=specification,
        model_type=model_type,
        tl_satisfaction_threshold=tl_satisfaction_threshold,
        detection_threshold=detection_threshold
    )

    frame_step = int(round(multi_video_data[0]["video_info"]["fps"] / multi_video_data[0]["sample_rate"])) # since they are identical, take from [0]
    frame_of_interest = FramesofInterest(num_of_frame_in_sequence, frame_step)

    multi_frames = [video_data["images"] for video_data in multi_video_data]

    frame_windows = []
    for i in range(0, len(multi_frames[0]), num_of_frame_in_sequence): # these are established to be the same length
        frame_windows.append([frames[i : i + num_of_frame_in_sequence] for frames in multi_frames])
    if PRINT_ALL:
        print(f"{len(frame_windows)} frame windows to process")
        print(f"{len(frame_windows[0])} cameras per frame window")
        print(f"{len(frame_windows[0][0])} frames per camera per window")
        print(f"{frame_windows[0][0][0].shape} shape of each frame")

    def process_frame(multi_sequence_of_frames: list[list[np.ndarray]], frame_count: int):
        object_of_interest = {}
        frame_images = {f"cam{i}": seq for i, seq in enumerate(multi_sequence_of_frames)}

        for prop in proposition:
            best_detection = (None, DetectedObject(name=prop, is_detected=False, confidence=0.0, probability=0.0))

            for cam_id, sequence_of_frames in frame_images.items():
                detected_object = vlm.detect(
                    seq_of_frames=sequence_of_frames,
                    scene_description=prop,
                    threshold=vlm_detection_threshold
                )
                if detected_object.confidence > best_detection[1].confidence:
                    best_detection = (cam_id, detected_object)

            object_of_interest[prop] = best_detection
            if PRINT_ALL and best_detection[1].is_detected:
                print(f"\t{prop} ({best_detection[0]}): {best_detection[1].confidence}->{best_detection[1].probability}")

        # print(frame_images.keys(), len(frame_images.values()))
        # print(object_of_interest)
        frame = VideoFrame(
            frame_idx=frame_count,
            frame_images=frame_images,
            object_of_interest=object_of_interest,
        )
        return frame

    if PRINT_ALL:
        looper = enumerate(frame_windows)
    else:
        looper = tqdm.tqdm(enumerate(frame_windows), total=len(frame_windows))

    all_detections = [set(), set()]
    for i, multi_sequence_of_frames in looper:
        if PRINT_ALL:
            print("\n" + "*"*50 + f" {i}/{len(frame_windows)-1} " + "*"*50)
            print(f"Detections:")
        frame = process_frame(multi_sequence_of_frames, i)
        if PRINT_ALL: # disabled
            os.makedirs(image_output_dir, exist_ok=True)
            frame.save_frame_img(save_path=os.path.join(image_output_dir, f"{i}"))

        if checker.validate_frame(frame_of_interest=frame):
            thresh = frame.thresholded_detected_objects(threshold=detection_threshold)
            for prop, (prob, cam_id) in thresh.items():
                split = checker.check_split(prop)
                if ((frame.frame_idx, cam_id) not in all_detections[split]):
                    all_detections[split].add((frame.frame_idx, cam_id))
            if PRINT_ALL:
                print(f"\t{[sorted(s) for s in all_detections]}")

            automaton.add_frame(frame=frame)
            frame_of_interest.frame_buffer.append(frame)
            model_check = checker.check_automaton(automaton=automaton)
            if model_check:
                automaton.reset()
                frame_of_interest.flush_frame_buffer()

    automaton_foi = frame_of_interest.compile_foi()
    if PRINT_ALL:
        print()
        print(f"Automaton indices: {automaton_foi}")

    if not automaton_foi: # automaton empty or nothing detected
        foi = {-1: {}}
    else:
        detections_with_cams = intersection_with_gaps(all_detections)
        if detections_with_cams:
            scaled_detections = {k * num_of_frame_in_sequence * frame_step: v for k, v in detections_with_cams.items()}
            min_frame = min(scaled_detections.keys())
            max_frame = max(scaled_detections.keys())
            detections_foi = list(range(int(min_frame), int(max_frame) + 1))

            if PRINT_ALL:
                print(f"Detection indices: {detections_foi}")

            intersecting_indices = sorted(set(automaton_foi) & set(detections_foi))

            if not intersecting_indices:
                foi = {-1: {}}
            else:
                start_frames = sorted(scaled_detections.keys())
                foi = {
                    frame: scaled_detections[start_frames[bisect.bisect_right(start_frames, frame) - 1]] for frame in intersecting_indices
                }
        else:
            foi = {-1: {}}

        if PRINT_ALL:
            print("\n" + "-"*107)
            print(f"All Detections: {all_detections}")
            print(f"Detected frames of interest:\n{foi}")

    return foi, all_detections

