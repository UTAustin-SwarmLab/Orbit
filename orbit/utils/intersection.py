def group_with_gaps(nums, max_gaps=2):
    groups = []
    current_group = [nums[0]]
    gaps = 0

    for i in range(1, len(nums)):
        if nums[i] - nums[i-1] == 1:
            current_group.append(nums[i])
        elif nums[i] - nums[i-1] <= max_gaps + 1:
            # fill in the gap logically
            current_group.extend(range(nums[i-1]+1, nums[i]+1))
            gaps += nums[i] - nums[i-1] - 1
            if gaps > max_gaps:
                groups.append(current_group[:- (nums[i] - nums[i-1] - 1)])
                current_group = [nums[i]]
                gaps = 0
        else:
            groups.append(current_group)
            current_group = [nums[i]]
            gaps = 0

    groups.append(current_group)
    return groups

def intersection_with_gaps(indices, max_gaps=1):
    if len(non_empty := [s for s in indices if s]) == 1:
        frame_indices = [idx for idx, cam in non_empty[0]]
        groups = group_with_gaps(frame_indices, max_gaps)
        largest_set = max(groups, key=len) if groups else []
        result = {}
        for frame_idx in largest_set:
            result[frame_idx] = [cam for idx, cam in non_empty[0] if idx == frame_idx]
        return result

    set_A, set_B = indices[0], indices[1]
    frame_indices_A = {idx for idx, cam in set_A}
    frame_indices_B = {idx for idx, cam in set_B}
    combined = sorted(list(frame_indices_A | frame_indices_B))

    largest_set = []
    for group in group_with_gaps(combined, max_gaps):
        if len(group) > len(largest_set):
            largest_set = group

    result = {}
    for frame_idx in largest_set:
        cams = []
        for idx, cam in set_A:
            if idx == frame_idx:
                cams.append(cam)
        for idx, cam in set_B:
            if idx == frame_idx:
                cams.append(cam)
        result[frame_idx] = list(set(cams))
    return result

