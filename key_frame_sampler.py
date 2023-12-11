import json
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


def calculate_difference_histogram(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff


def calculate_difference_feature_points(frame1, frame2):
    """ORB feature points"""
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据匹配的数量计算差异
    diff = len(matches)
    return diff


def calculate_difference_optical_flow(frame1, frame2):
    """optical flow"""
    # gray image
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # average motion
    diff = np.mean(magnitude)
    return diff


def calculate_difference_deep_learning(frame1, frame2):
    """dl"""
    model = models.resnet18(pretrained=True).eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # extract features
    input1 = preprocess(frame1).unsqueeze(0)
    input2 = preprocess(frame2).unsqueeze(0)

    with torch.no_grad():
        features1 = model(input1)
        features2 = model(input2)

    # norm
    diff = torch.norm(features1 - features2).item()
    return diff


def calculate_frame_difference(frame1, frame2):
    """simple difference"""
    diff = cv2.absdiff(frame1, frame2)
    non_zero_count = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
    return non_zero_count


def calculate_threshold(frame_differences, max_interval, beta=1):
    return beta * np.mean(frame_differences) * max_interval


def extract_key_frames(input_folder, cal_diff_func, max_interval=10, total_frames=-1, alpha=1):
    """Adaptive frame sampling extracts key frames"""

    frames = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder)])
    if total_frames != -1:
        frames = frames[:total_frames]

    print(cal_diff_func)
    if cal_diff_func == "hist":
        calculate_difference_function = calculate_difference_histogram
    elif cal_diff_func == "feature":
        calculate_difference_function = calculate_difference_feature_points
    elif cal_diff_func == "optical":
        calculate_difference_function = calculate_difference_optical_flow
    elif cal_diff_func == "dl":
        calculate_difference_function = calculate_difference_deep_learning
    elif cal_diff_func == "abs":
        calculate_difference_function = calculate_frame_difference
    else:
        print("!!Base!!")
        return list(range(0, len(frames), max_interval))

    frame_differences = []
    last_frame = None

    # calculate difference
    for frame_path in tqdm(frames):
        frame = cv2.imread(frame_path)
        if last_frame is not None:
            difference = calculate_difference_function(last_frame, frame)
            frame_differences.append(difference)
        last_frame = frame

    # threshold
    threshold = calculate_threshold(frame_differences, max_interval)
    last_key_frame_index = -1

    key_frame_idxes = set()

    # extract key frames
    accumulated_diff = 0
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if i > 0:
            accumulated_diff += frame_differences[i - 1]

        if (i == 0 or (i - last_key_frame_index) >= alpha * max_interval
                or (accumulated_diff > threshold and i - last_key_frame_index > 1)):
            # key_frame_path = os.path.join(output_folder, f'key_frame_{i}.jpg')
            # cv2.imwrite(key_frame_path, frame)
            key_frame_idxes.add(i)
            last_key_frame_index = i
            accumulated_diff = 0  # 重置累积差异

    key_frame_idxes.add(last_key_frame_index)
    key_frame_idxes = sorted(list(key_frame_idxes))

    print(len(key_frame_idxes), key_frame_idxes)

    return key_frame_idxes


# def extract_key_frames(input_folder, cal_diff_func, max_interval=10, total_frames=-1):
#     """Extracts key frames based on a regular interval or difference-based selection"""
#
#     frames = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder)])
#     if total_frames == -1:
#         total_frames = len(frames)
#     else:
#         frames = frames[:total_frames]
#     print(total_frames)
#
#     num_key_frames = total_frames // max_interval + (1 if total_frames % max_interval else 0)
#
#     # Method 1: Uniform Sampling
#     uniform_key_frames = [frames[i * max_interval] for i in range(num_key_frames)]
#     uniform_key_frames[-1] = frames[-1]  # Ensure last frame is included
#
#     # Method 2: Difference-Based Sampling (Using given difference calculation method)
#     print(cal_diff_func)
#     if cal_diff_func == "hist":
#         calculate_difference_function = calculate_difference_histogram
#     elif cal_diff_func == "feature":
#         calculate_difference_function = calculate_difference_feature_points
#     elif cal_diff_func == "optical":
#         calculate_difference_function = calculate_difference_optical_flow
#     elif cal_diff_func == "dl":
#         calculate_difference_function = calculate_difference_deep_learning
#     elif cal_diff_func == "abs":
#         calculate_difference_function = calculate_frame_difference
#     else:
#         print("!!Base!!")
#         return list(range(0, len(frames), max_interval))
#
#     # num key frames
#     expected_num_intervals = (total_frames - 1) // max_interval
#
#     # frame differences
#     frame_differences = []
#     for i in range(1, total_frames):
#         frame = cv2.imread(frames[i])
#         last_frame = cv2.imread(frames[i - 1])
#         difference = calculate_difference_function(last_frame, frame)
#         frame_differences.append(difference)
#
#     # target
#     total_difference = sum(frame_differences)
#     target_diff_per_interval = total_difference / expected_num_intervals
#
#     # select
#     key_frame_idxes = {0}
#     accumulated_diff = 0
#
#     for i in range(1, total_frames):
#         accumulated_diff += frame_differences[i - 1]
#         # sum > target sum
#         if accumulated_diff >= target_diff_per_interval:
#             key_frame_idxes.add(i)
#             accumulated_diff = 0
#
#     # last
#     # key_frame_idxes.add(len(frames) - 1)
#     key_frame_idxes = sorted(list(key_frame_idxes))
#
#     print(key_frame_idxes, len(key_frame_idxes))
#     return key_frame_idxes


def main():

    # work_name = "woman"
    work_name = "cxk"
    print(work_name)

    output_folder = f'./videos/{work_name}/keyframes'
    os.makedirs(output_folder, exist_ok=True)

    for func in ["hist", "feature", "optical", "dl", "base"]:
        key_frame_idxes = extract_key_frames(f'./videos/{work_name}/video', func)
        with open(os.path.join(output_folder, func + ".json"), "w") as f:
            json.dump(key_frame_idxes, f, indent=4)


if __name__ == '__main__':
    main()
