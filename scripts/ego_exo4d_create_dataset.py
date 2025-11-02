from orbit.puls.llm import *
from openai import OpenAI

import random
import glob
import json
import tqdm
import os
import re

DATASET_PATH = "/nas/mars/dataset/Ego-Exo4D"
NARRATIONS_PATH = "outputs/ego_exo4d_narrations.json"
VQA_FIRST_PASS_PATH = "outputs/ego_exo4d_vqa_1.json"
VQA_SECOND_PASS_PATH = "outputs/ego_exo4d_vqa_2.json"
VQA_FINAL_DATASET_PATH = "outputs/ego_exo4d_dataset.json"
NUM_PER_CATEGORY = 6

def create_prompt_0(narrations):
    full_prompt =  f"""
You are an AI assistant specializing in creating assessment materials. Your task is to generate multiple-choice questions (MCQs) based on a provided list of chronological narrations. The narrations describe a sequence of events in the exact order they occurred.

Your goal is to create five complex multiple-choice questions that test a user's understanding of the sequence of events.

Here is the chronological list of narrations:

{narrations}

## Question Generation Rules

Please adhere to the following rules for generating the questions and answers:

### 1. Question Style
* Most questions should be "What happened **after**..." or "What happened **before**..."
* Include at least two "complex" questions that refer to two consecutive events.
    * Example: "What happened *after* the subject [Action X] *and then* [Action Y]?"

### 2. Answer Format
* Provide four multiple-choice options (a, b, c, d).
* Clearly indicate the correct answer for each question in the format `Correct Answer: [letter]`.

### 3. Correct Answer Rules
* For "What happened **after** X?" questions, the correct answer **must** be an event that occurs 1-3 steps *after* event X in the narration list.
* For "What happened **before** X?" questions, the correct answer **must** be an event that occurs 1-3 steps *before* event X in the narration list.
* The correct answer must be a direct quote or a very close paraphrase of an event from the list.

### 4. Distractor (Incorrect Answer) Rules
You now have three plausible but incorrect distractors for each question. Use the following strategies:

* **Time-Reversed:** For an "after" question, use an event that happened *before* the event in the question. For a "before" question, use an event that happened *after*.
* **Opposite End:** Use an event from the *opposite end* of the narration list.
    * If the question is about an event at the beginning (e.g., "lifts a bicycle wheel"), use a distractor from the very end (e.g., "places the wrench in the tool container").
    * If the question is about an event at the end (e.g., "tightens the wheel axle nut"), use a distractor from the very beginning (e.g., "looses the valve nut").
* **Negations/Opposites:** Create an answer choice that is the *opposite* of an action in the list.
    * Example: If the list says "Subject turns the wheel," a distractor could be "Subject *drops* the wheel" or "Subject *stops touching* the wheel."
    * Example: "Subject *fails* to tighten the valve nut."

---

## Desired Output Examples

*Please generate your 5 new questions in this exact format.*

1. What happened shortly after the subject picks a pump from the pump hanger with his right hand?
a) Subject places the wrench in the tool container with his right hand.
b) Subject looses the valve nut of the wheel in his left hand with his right hand.
c) Subject drops the pump on the floor.
d) Subject turns the head of the pump in his right hand with his left hand.
Correct Answer: d

2. After the subject lifts the wheel from the floor with his left hand and then holds it with both hands, what happens next?
a) Subject holds the wheel on the floor with his left hand.
b) Subject positions the wheel between the front fork of the bicycle with both hands.
c) Subject places the wheel back on the floor.
d) Subject picks a pump from the pump hanger with his right hand.
Correct Answer: b

3. What happened right before the subject passes the wrench from his right hand to his left hand (Event 53)?
a) Subject tightens the wheel axle nut with the wrench in his right hand.
b) Subject holds the wheel with his right hand.
c) Subject lifts a bicycle wheel from the floor with his left hand.
d) Subject drops the wrench.
Correct Answer: a
"""
    return full_prompt

def create_prompt_2(narrations, question):
    full_prompt =  f"""
You are a precise, analytical AI assistant. Your sole purpose is to act as an answer key for a multiple-choice question based on a provided chronological list of narrations.

## Instructions
1.  Carefully read the **Chronological Narrations** to understand the exact sequence of events.
2.  Read the **Question** and analyze what it is asking (e.g., "What happened after X?", "What happened before Y?").
3.  Locate the event(s) mentioned in the question within the narration list.
4.  Examine each answer choice (a, b, c, d) and compare it to the narration list to find the one that factually answers the question.
    * For "after" questions, the correct answer must be an event that occurs *after* the event in the question.
    * For "before" questions, the correct answer must be an event that occurs *before* the event in the question.
5.  Your output MUST be a single letter (a, b, c, or d) and nothing else. Do not provide explanation, punctuation, or any other text.

---

## Chronological Narrations
{narrations}

---

## Question
{question}
"""
    return full_prompt

def get_unique_video_types():
    videos_path = os.path.join(DATASET_PATH, "takes")
    all_entries = os.listdir(videos_path)
    dir_names = [d for d in all_entries if os.path.isdir(os.path.join(videos_path, d))]
    category_names = [re.split(r"[_]?\d+", d)[0] for d in dir_names]
    unique_categories = sorted(list(set(category_names)))
    return unique_categories

def read_narrations_map():
    annotations = {}
    file_names = ["atomic_descriptions_train.json", "atomic_descriptions_val.json"]
    for file_name in file_names:
        with open(os.path.join(DATASET_PATH, "annotations", file_name), "r") as f:
            annotations.update(json.load(f)["annotations"])
    return annotations

def task_uid_map():
    with open(os.path.join(DATASET_PATH, "takes.json"), "r") as f:
        takes = json.load(f)
    name_to_uid = {}
    for take in takes:
        name_to_uid[take["take_name"]] = take["take_uid"]
    return name_to_uid

def get_uids_with_atomic_descriptions():
    with open(os.path.join(DATASET_PATH, "annotations", "splits.json"), "r") as f:
        splits = json.load(f)["take_uid_to_benchmark"]
    ret = set()
    for k, v in splits.items():
        if "atomic_action_descriptions" in v:
            ret.add(k)
    return ret

def create_narrations():
    unique_types = get_unique_video_types()
    all_videos = sorted([d for d in os.listdir(os.path.join(DATASET_PATH, "takes"))])

    narrations_map = read_narrations_map()

    name_to_uid = task_uid_map()

    atomic_description_uids = get_uids_with_atomic_descriptions()

    main_dict = {}
    # 1. iterate through video types
    for video_type in unique_types:
        videos_in_category = [v for v in all_videos if v.startswith(video_type)]
        total_added = 0
        # 2. iterate through all videos in that type
        for video in videos_in_category:
            # 3. convert video name to uid
            uid = name_to_uid[video]
            # 4. check if uid has atomic descriptions
            if uid in atomic_description_uids:
                # 5. add to main dict if one caption exists
                if uid in narrations_map and len(narrations_map[uid]) == 1:
                    total_added += 1
                    mp4_files = glob.glob(os.path.join(DATASET_PATH, "takes", video, "frame_aligned_videos", "downscaled", "448", "cam*.mp4"))
                    narrations = narrations_map[uid][0]["descriptions"]
                    modified_narrations = []
                    for narration in narrations:
                        modified_text = re.sub(r"\bC\b", "Camera Wearer", narration["text"])
                        modified_text = re.sub(r"\bO\b", "Other Person", modified_text)
                        modified_narrations.append(modified_text)
                    main_dict[video] = {
                        "video_paths": sorted(mp4_files),
                        "narrations": modified_narrations
                    }
            # 6. break once amount per category criteria reached
            if total_added >= NUM_PER_CATEGORY:
                break

    with open(NARRATIONS_PATH, "w") as f:
        json.dump(main_dict, f, indent=4)

def parse_through_openai():
    with open(NARRATIONS_PATH, "r") as f:
        raw_data = json.load(f)

    output = {}
    for key in tqdm.tqdm(raw_data):
        narration = raw_data[key]["narrations"]

        client = OpenAI()
        llm = LLM(client)

        prompt = create_prompt_0(narration)
        llm_output = llm.prompt(prompt)
        output[key] = llm_output

    with open(VQA_FIRST_PASS_PATH, "w") as f:
        json.dump(output, f, indent=4)

def check_with_reprompt():
    with open(VQA_FIRST_PASS_PATH, "r") as f:
        questions_data = json.load(f)
    with open(NARRATIONS_PATH, "r") as f:
        narrations_data = json.load(f)
    
    good_questions = {}
    bar = tqdm.tqdm(narrations_data)
    for key in bar:
        narrations = narrations_data[key]["narrations"]
        question_strings = questions_data[key].split("\n\n")

        parsed_questions = []
        for q_str in question_strings:
            parts = q_str.strip().split("Correct Answer:")
            if len(parts) == 2:
                parsed_questions.append((parts[0].strip(), parts[1].strip()))

        possible_choices = []
        for question, correct_answer in parsed_questions:
            client = OpenAI()
            llm = LLM(client)

            prompt  = create_prompt_2(narrations, question)
            llm_output = llm.prompt(prompt).strip()
            if correct_answer == llm_output:
                possible_choices.append((question, correct_answer))
        good_questions[key] = possible_choices
        bar.set_description(f"({len(possible_choices)}/{len(question_strings)})")

    with open(VQA_SECOND_PASS_PATH, "w") as f:
        json.dump(good_questions, f, indent=4)

def output_final_json():
    with open(VQA_SECOND_PASS_PATH, "r") as f:
        questions_data = json.load(f)
    with open(NARRATIONS_PATH, "r") as f:
        narrations_data = json.load(f)

    final_output = {}
    for key in narrations_data:
        if questions_data[key] != []:
            qa_pair = random.choice(questions_data[key])
            question_candidates = qa_pair[0][3:].split("\n")
            question = question_candidates[0].strip()
            candidates = [c.strip() for c in question_candidates[1:]]
            final_output[key] = {
                "video_paths": narrations_data[key]["video_paths"],
                "question": question,
                "candidates": candidates,
                "correct_answer": qa_pair[1]
            }
    with open(VQA_FINAL_DATASET_PATH, "w") as f:
        json.dump(final_output, f, indent=4)

def main():
    # print("--- Creating Narrations ---")
    # create_narrations()
    # print("\n--- Parsing through OpenAI ---")
    # parse_through_openai()
    # print("\n--- Checking with Reprompt ---")
    # check_with_reprompt()
    print("\n--- Outputting Final JSON ---")
    output_final_json()

if __name__ == "__main__":
    main()

