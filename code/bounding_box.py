import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 이미지 경로 설정
image_paths = [
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy60.png",
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy.png",
]

# YOLO 모델 결과에 맞는 바운딩박스 좌표
bounding_boxes = [
    {"class": "sofa", "coords": [850, 700, 1050, 1000]},  # 첫 번째 이미지
    {"class": "sofa", "coords": [850, 550, 1050, 900]},  # 두 번째 이미지
]

# 이미지 크기 설정
image_width = 640
image_height = 640

# 라벨 파일을 저장할 폴더
label_folder = "/Users/jiwan/Desktop/yolo_fl/dataset/labels/Train/"
os.makedirs(label_folder, exist_ok=True)

# 이미지와 라벨 파일 생성
for i, image_path in enumerate(image_paths):
    # 이미지 로드
    img = Image.open(image_path).convert('RGB')
    
    # 바운딩박스 좌표 추출
    bbox = bounding_boxes[i]["coords"]
    x_min, y_min, x_max, y_max = bbox
    
    # 바운딩박스를 YOLO 형식으로 변환 (정규화)
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
    # 정규화된 좌표가 0 ~ 1 범위 내로 유지되도록 (경계 초과 방지)
    x_center = min(max(x_center, 0), 1)
    y_center = min(max(y_center, 0), 1)
    width = min(max(width, 0), 1)
    height = min(max(height, 0), 1)
    
    # 라벨 파일 경로 설정
    label_file_path = os.path.join(label_folder, f"bounding_box_{i + 1}.txt")
    
    # 클래스 ID (소파는 0번)
    class_id = 0
    
    # YOLO 형식으로 바운딩박스 라벨 작성
    with open(label_file_path, "w") as label_file:
        label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 바운딩박스를 이미지에 그리기
    fig, ax = plt.subplots()
    ax.imshow(img)

    # 바운딩박스 그리기
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()

print("✅ 바운딩박스가 추가된 이미지를 저장하고 라벨 파일 생성 완료!")
