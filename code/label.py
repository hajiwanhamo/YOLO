import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# 4개의 이미지 파일 경로
image_paths = [
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy60.png",
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy.png",
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image.png"
]

# YOLO 모델 결과를 반영한 바운딩박스 좌표 (예시)
bounding_boxes = [
    {"class": "sofa", "coords": [0.403, 0.416, 0.069, 0.173]},  # 첫 번째 이미지
    {"class": "sofa", "coords": [0.333, 0.333, 0.089, 0.255]},  # 두 번째 이미지
    {"class": "sofa", "coords": [0.556, 0.444, 0.125, 0.278]}   # 세 번째 이미지
]

image_width = 1440
image_height = 1440

# 각각의 이미지에 대해 다른 이동 값을 설정
move_x_values = [380, 480, 50]  # 각각의 이미지에 대해 이동할 값 (픽셀)
move_y_values = [200, 250, 50]  # 각각의 이미지에 대해 이동할 y 값 (픽셀)

# 라벨 파일을 저장할 폴더 경로 설정
label_folder = "/Users/jiwan/Desktop/yolo_fl/dataset/labels/Train/"
os.makedirs(label_folder, exist_ok=True)

# 3개의 이미지를 한 번에 표시하기 위한 설정 (각각 이미지 저장 및 라벨 생성)
for i, image_path in enumerate(image_paths):
    fig, ax = plt.subplots(figsize=(12, 12))
    img = Image.open(image_path).convert('RGB')
    ax.imshow(img)

    # YOLO 형식 바운딩박스를 픽셀 좌표로 변환
    bbox = bounding_boxes[i]["coords"]
    x_center, y_center, width, height = bbox
    
    # 픽셀 좌표로 변환
    x_min = (x_center - width / 2) * image_width
    x_max = (x_center + width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    y_max = (y_center + height / 2) * image_height

    # 바운딩박스를 이동시킬 값 (각각 다르게 설정)
    move_x = move_x_values[i]  # 해당 이미지에 맞는 이동값
    move_y = move_y_values[i]  # 해당 이미지에 맞는 이동값
    
    x_min += move_x
    x_max += move_x
    y_min += move_y
    y_max += move_y
    
    # 바운딩박스 그리기
    ax.add_patch(plt.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        edgecolor='red', facecolor='none', linewidth=2
    ))

    # 클래스 이름 표시
    ax.text(x_min, y_min - 10, bounding_boxes[i]["class"], color='red', fontsize=12)

    ax.axis('off')  # 축을 표시하지 않음

    # 이미지 저장
    save_image_path = f"/Users/jiwan/Desktop/yolo_fl/dataset/images/Train/bounding_box_{i + 1}.png"
    plt.savefig(save_image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 다음 이미지를 위해 플롯을 닫음

    # 라벨 파일 생성 (YOLO 형식)
    label_file_path = os.path.join(label_folder, f"bounding_box_{i + 1}.txt")
    class_id = 0  # 'sofa' 클래스는 ID 0으로 설정

    # 바운딩박스를 YOLO 형식으로 변환 (0~1 범위로 정규화)
    x_center_normalized = (x_center * image_width) / image_width
    y_center_normalized = (y_center * image_height) / image_height
    width_normalized = (width * image_width) / image_width
    height_normalized = (height * image_height) / image_height

    # 라벨 파일에 바운딩박스 정보 작성
    with open(label_file_path, "w") as label_file:
        label_file.write(f"{class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n")

print("✅ 바운딩박스가 추가된 이미지를 저장하고 라벨 파일 생성 완료!")
