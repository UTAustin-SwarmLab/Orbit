from orbit.datamanager.manager import Manager

from collections import defaultdict
from tqdm import tqdm
import hashlib
import shutil
import json
import copy
import os


class EgoExo4D(Manager):
    def __init__(self):
        self.compile_position = False
        self.compile_full = True

        self._dataset_path = "/nas/mars/dataset/Ego-Exo4D"
        self._question_path = "/nas/mars/experiment_result/orbit/1_dataset_json/ego_exo4d_dataset.json"
        
    def load_data(self):
        with open(self._question_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        ret = []
        for key in dataset:
            ret.append({
                "question": dataset[key]["question"],
                "candidates": dataset[key]["candidates"],
                "correct_answer": dataset[key]["correct_answer"],
                "video_paths": dataset[key]["video_paths"],
                "video_id": key
            })
        return ret


    def postprocess_data(self, nsvs_path):
        self._nsvs_path = nsvs_path
        run_name = self._nsvs_path.split('/')[-1].split('.')[0].replace('longvideobench_', '')
        self._output_path_nsvqa = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_nsvqa_{run_name}"
        if self.compile_position:
            self._output_path_position = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_position_{run_name}"
        if self.compile_full:
            self._output_path_full = f"/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_full_{run_name}"

        os.makedirs(os.path.join(self._output_path_nsvqa, "videos"), exist_ok=True)
        if self.compile_position:
            os.makedirs(os.path.join(self._output_path_position, "videos"), exist_ok=True)
        if self.compile_full:
            os.makedirs(os.path.join(self._output_path_full, "videos"), exist_ok=True)
            shutil.copytree("/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_full/video", os.path.join(self._output_path_full, "videos"), dirs_exist_ok=True)

        with open(os.path.join(self._dataset_path, "lvb_val.json"), "r") as f:
            lvb_data = json.load(f)
        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output_nsvqa = []    # nsvqa cropped video
        output_full = []     # entire video
        for entry_nsvs in tqdm(nsvs_data):
            found = False
            for entry in lvb_data:
                if entry["question"] == entry_nsvs["question"] and entry["id"] == entry_nsvs["metadata"]["id"]:
                    found = True

                    candidates = entry["candidates"]
                    for i in range(5):
                        if i < len(candidates):
                            entry[f"option{i}"] = candidates[i]
                        else:
                            entry[f"option{i}"] = "N/A"

                    entry_full = copy.deepcopy(entry)

                    code = entry["question"] + entry["id"]
                    id = hashlib.sha256(code.encode()).hexdigest()
                    entry["id"] = id + "_0"
                    entry["video_id"] = id
                    entry["video_path"] = id + ".mp4"


                    self.crop_video(
                        entry_nsvs, 
                        save_path=os.path.join(self._output_path_nsvqa, "videos", entry["video_path"]),
                        ground_truth=False
                    )

                    if os.path.exists(os.path.join(self._output_path_nsvqa, "videos", entry["video_path"])): # if crop is successful
                        if self.compile_position:
                            self.crop_video(
                                entry_nsvs, 
                                save_path=os.path.join(self._output_path_position, "videos", entry["video_path"]),
                                ground_truth=True
                            )

                        output_nsvqa.append(entry)
                        output_full.append(entry_full)

            if found == False:
                print(f"Entry not found for question: {entry_nsvs['question']}")

        with open(os.path.join(self._output_path_nsvqa, "lvb_val.json"), "w") as f:
            json.dump(output_nsvqa, f, indent=4)
        if self.compile_position:
            with open(os.path.join(self._output_path_position, "lvb_val.json"), "w") as f:
                json.dump(output_nsvqa, f, indent=4)
        if self.compile_full:
            with open(os.path.join(self._output_path_full, "lvb_val.json"), "w") as f:
                json.dump(output_full, f, indent=4)

