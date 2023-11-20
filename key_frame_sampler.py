import json
import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm


def calculate_difference_histogram(frame1, frame2):
    """使用直方图比较计算两帧之间的差异"""
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return diff


def calculate_difference_feature_points(frame1, frame2):
    """使用ORB特征点匹配计算两帧之间的差异"""
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(frame1, None)
    kp2, des2 = orb.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据匹配的数量计算差异
    diff = len(matches)
    return diff


def calculate_difference_optical_flow(frame1, frame2):
    """使用光流法计算两帧之间的差异"""
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 使用运动的平均幅度作为差异
    diff = np.mean(magnitude)
    return diff


def calculate_difference_deep_learning(frame1, frame2):
    """使用深度学习模型提取特征计算两帧之间的差异"""
    # 加载预训练模型
    model = models.resnet18(pretrained=True).eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 提取特征
    input1 = preprocess(frame1).unsqueeze(0)
    input2 = preprocess(frame2).unsqueeze(0)

    with torch.no_grad():
        features1 = model(input1)
        features2 = model(input2)

    # 计算特征之间的欧几里得距离
    diff = torch.norm(features1 - features2).item()
    return diff


def calculate_frame_difference(frame1, frame2):
    """计算两帧之间的差异，这里使用简单的像素差异"""
    diff = cv2.absdiff(frame1, frame2)
    non_zero_count = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
    return non_zero_count


def calculate_threshold(frame_differences, max_interval):
    """根据所有帧的差异来计算阈值"""
    # 可以根据需要调整这里的计算方式，例如使用平均值或总和
    return np.mean(frame_differences) * max_interval


def extract_key_frames(input_folder, cal_diff_func, max_interval=10):
    """自适应帧采样提取关键帧"""

    frames = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder)])

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

    # 首先计算所有帧之间的差异
    for frame_path in tqdm(frames):
        frame = cv2.imread(frame_path)
        if last_frame is not None:
            difference = calculate_difference_function(last_frame, frame)
            frame_differences.append(difference)
        last_frame = frame

    # 计算阈值
    threshold = calculate_threshold(frame_differences, max_interval)
    last_key_frame_index = -1

    key_frame_idxes = set()

    # 提取关键帧
    # 提取关键帧
    accumulated_diff = 0
    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if i > 0:
            accumulated_diff += frame_differences[i - 1]

        if i == 0 or (i - last_key_frame_index) >= max_interval or accumulated_diff > threshold:
            # key_frame_path = os.path.join(output_folder, f'key_frame_{i}.jpg')
            # cv2.imwrite(key_frame_path, frame)
            key_frame_idxes.add(i)
            last_key_frame_index = i
            accumulated_diff = 0  # 重置累积差异

    key_frame_idxes.add(last_key_frame_index)
    key_frame_idxes = sorted(list(key_frame_idxes))

    return key_frame_idxes


def main():
    output_folder = f'./videos/woman/keyframes'
    os.makedirs(output_folder, exist_ok=True)

    for func in ["hist", "feature", "optical", "dl", "base"]:
        key_frame_idxes = extract_key_frames('./videos/woman/video', func)
        with open(os.path.join(output_folder, func + ".json"), "w") as f:
            json.dump(key_frame_idxes, f, indent=4)




