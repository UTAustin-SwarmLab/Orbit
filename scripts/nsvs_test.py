from itertools import groupby

from orbit.nsvs.model_checker.property_checker import PropertyChecker
from orbit.nsvs.model_checker.video_automaton import VideoAutomaton, VideoFrame
from orbit.nsvs.video.frames_of_interest import FramesofInterest
from orbit.nsvs.vlm.obj import DetectedObject


def main():
    proposition = ["a", "b", "c"]
    specification_raw = '("a" & "b") U "c"'
    data = [
        {"a": 0, "b": 0, "c": 0},  # 0   ___
        {"a": 0, "b": 0, "c": 0},  # 1   ___
        {"a": 0, "b": 0, "c": 0},  # 2   ___
        {"a": 0, "b": 0, "c": 0},  # 3   ___
        {"a": 1, "b": 0, "c": 0},  # 4   a__
        {"a": 1, "b": 1, "c": 0},  # 5   ab_
        {"a": 1, "b": 1, "c": 0},  # 6   ab_
        {"a": 1, "b": 1, "c": 0},  # 7   ab_
        {"a": 1, "b": 1, "c": 0},  # 8   ab_
        {"a": 0, "b": 0, "c": 0},  # 9   ___
        {"a": 0, "b": 0, "c": 0},  # 10  ___
        {"a": 0, "b": 0, "c": 0},  # 11  ___
        {"a": 0, "b": 0, "c": 0},  # 12  ___
        {"a": 1, "b": 1, "c": 0},  # 13  ab_
        {"a": 1, "b": 1, "c": 0},  # 14  ab_
        {"a": 1, "b": 1, "c": 0},  # 15  ab_
        {"a": 1, "b": 1, "c": 0},  # 16  ab_
        {"a": 1, "b": 1, "c": 1},  # 17  abc
        {"a": 1, "b": 1, "c": 1},  # 18  abc
        {"a": 1, "b": 1, "c": 1},  # 19  abc
        {"a": 1, "b": 1, "c": 1},  # 20  abc
        {"a": 1, "b": 1, "c": 1},  # 21  abc
        {"a": 0, "b": 0, "c": 1},  # 22  __c
        {"a": 0, "b": 0, "c": 1},  # 23  __c
        {"a": 0, "b": 0, "c": 1},  # 24  __c
        {"a": 0, "b": 0, "c": 1},  # 25  __c
        {"a": 0, "b": 0, "c": 0},  # 26  ___
        {"a": 0, "b": 0, "c": 0},  # 27  ___
        {"a": 0, "b": 0, "c": 0},  # 28  ___
        {"a": 0, "b": 0, "c": 0},  # 29  ___
    ]
    # data = [
    #     {'a': 0, 'b': 0, 'c': 0}, # 0:  ___
    #     {'a': 1, 'b': 1, 'c': 0}, # 1:  ab_
    #     {'a': 0, 'b': 0, 'c': 0}, # 2:  ___
    #     {'a': 1, 'b': 1, 'c': 0}, # 3:  ab_
    #     {'a': 1, 'b': 1, 'c': 1}, # 4:  abc
    #     {'a': 0, 'b': 0, 'c': 1}, # 5:  __c
    #     {'a': 0, 'b': 0, 'c': 0}  # 6:  ___
    # ]
    automaton = VideoAutomaton(include_initial_state=True)
    automaton.set_up(proposition_set=proposition)
    checker_a = PropertyChecker(proposition=proposition, specification=specification_raw.split("U")[0])
    checker_b = PropertyChecker(proposition=proposition, specification=specification_raw.split("U")[1])
    frame_of_interest = FramesofInterest()

    for i in range(len(data)):
        print()
        object_of_interest = {}
        for detection in data[i]:
            object_of_interest[detection] = DetectedObject(
                name=detection,
                is_detected=(data[i][detection] > 0.50),
                confidence=data[i][detection],
                probability=data[i][detection],
            )
        frame = VideoFrame(frame_idx=i, frame_images=[], object_of_interest=object_of_interest)
        print(f"Index {i}  --  {frame.thresholded_detected_objects(threshold=0.50)}")

        if checker_a.validate_frame(frame_of_interest=frame):
            automaton.add_frame(frame=frame, checker=0)
            model_check = checker_a.check_automaton(automaton=automaton)
            if model_check:
                automaton.reset()

        if checker_b.validate_frame(frame_of_interest=frame):
            automaton.add_frame(frame=frame, checker=1)
            model_check = checker_b.check_automaton(automaton=automaton)
            if model_check:
                automaton.reset()

    print(f"Indices: {process_logic(automaton.indices)}")


def process_logic(indices):
    A = set(indices[0])
    B = set(indices[1])
    combined = sorted(A | B)

    def group_consecutive(nums):
        groups = []
        for _, g in groupby(enumerate(nums), key=lambda x: x[0] - x[1]):
            groups.append([x[1] for x in g])
        return groups

    result = []
    for group in group_consecutive(combined):
        if group and group[0] in A and group[-1] in B:
            result = group
            break
    print(result)


if __name__ == "__main__":
    main()
