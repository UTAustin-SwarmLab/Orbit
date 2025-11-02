from abc import ABC, abstractmethod
import subprocess
import json
import cv2

class Manager(ABC):
    @abstractmethod
    def load_data(self) -> list:
        pass
    
    @abstractmethod
    def postprocess_data(self, nsvs_path):
        pass

    def crop_video(self, entry, save_path):
        def group_into_ranges(frames):
            if not frames:
                return []
            frames = sorted(set(frames))
            ranges = []
            start = prev = frames[0]
            for f in frames[1:]:
                if f == prev + 1:
                    prev = f
                else:
                    ranges.append((start, prev + 1))  # ffmpeg uses end-exclusive
                    start = prev = f
            ranges.append((start, prev + 1))
            return ranges

        input_path = entry["paths"]["video_path"]

        # Step 1: Determine frames to keep
        frame_list = entry.get("frames_of_interest", [])
        if entry.get("nsvs", {}).get("output") == [-1]:
            # Get total frame count
            cap = cv2.VideoCapture(input_path)
            frame_list = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            cap.release()
        elif frame_list:
            start, end = min(frame_list), max(frame_list)
            frame_list = list(range(start, end + 1))

        # Fallback in case frame_list is still empty
        if not frame_list:
            cap = cv2.VideoCapture(input_path)
            frame_list = list(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            cap.release()

        ranges = group_into_ranges(frame_list)

        if not ranges:
            print(f"[Warning] No valid ranges for {input_path}, skipping.")
            return

        # Build filter_complex and output map
        filters = []
        labels = []
        for i, (start, end) in enumerate(ranges):
            filters.append(
                f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v{i}]"
            )
            labels.append(f"[v{i}]")
        filters.append(f"{''.join(labels)}concat=n={len(ranges)}:v=1[outv]")

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", "; ".join(filters),
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            save_path
        ]

        subprocess.run(cmd, check=True)

