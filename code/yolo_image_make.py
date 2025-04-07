import os
from PIL import Image
import albumentations as A
import numpy as np

# 이미지 파일 경로
image_paths = [
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy60.png",
    "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy.png",
]

# YOLO 모델 결과를 반영한 바운딩박스 좌표 (정답 설정)
bounding_boxes = [
    {"class": "sofa", "coords": [850, 700, 1050, 1000]},  # 첫 번째 이미지
    {"class": "sofa", "coords": [850, 550, 1050, 900]},  # 두 번째 이미지
]

# 이미지 크기 설정 (2400 x 1800)
image_width = 2400
image_height = 1800

# 라벨 파일을 저장할 폴더 경로
label_folder = "/Users/jiwan/Desktop/yolo_fl/dataset/labels/Train/"
os.makedirs(label_folder, exist_ok=True)

# 데이터 증강을 위한 설정 (이동 및 크기 변경을 하지 않는 증강만 사용)
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),  # 밝기/대비 조정
    A.HueSaturationValue(p=0.2),  # 색상, 채도, 밝기 변화
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# 이미지와 라벨 파일 생성
for i, image_path in enumerate(image_paths):
    # 이미지 로드
    img = Image.open(image_path).convert('RGB')

    # 바운딩박스 좌표
    bbox = bounding_boxes[i]["coords"]
    x_min, y_min, x_max, y_max = bbox

    # 바운딩박스를 픽셀로 변환
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    # 바운딩박스를 YOLO 형식으로 정규화
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height

    # 라벨 파일 경로 설정
    label_file_path = os.path.join(label_folder, f"bounding_box_{i + 1}.txt")

    # 클래스 ID (sofa는 0번)
    class_id = 0

    # YOLO 형식으로 바운딩박스를 라벨 파일에 저장
    with open(label_file_path, "w") as label_file:
        label_file.write(f"{class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n")

    # 이미지를 증강하여 20번 저장
    for j in range(20):  # 20번 반복하여 증강된 이미지 생성
        img_np = np.array(img)  # PIL 이미지를 NumPy 배열로 변환

        # 'bboxes'를 NumPy 배열로 변환
        bboxes = np.array([[x_min, y_min, x_max, y_max]])

        # 알버멘테이션 수행 (이동이나 회전 없이 밝기, 채도 등만 변환)
        augmented = transform(image=img_np, bboxes=bboxes, class_labels=['sofa'])
        augmented_img = augmented['image']
        bboxes = augmented['bboxes']

        # 증강된 이미지를 다시 PIL 형식으로 변환
        augmented_pil_img = Image.fromarray(augmented_img)

        # 증강된 이미지를 저장
        save_image_path = f"/Users/jiwan/Desktop/yolo_fl/dataset/images/Train/bounding_box_aug_{i + 1}_{j + 1}.png"
        augmented_pil_img.save(save_image_path)

        # 바운딩박스 라벨 파일 생성 (YOLO 형식)
        label_file_path = os.path.join(label_folder, f"bounding_box_aug_{i + 1}_{j + 1}.txt")
        
        # YOLO 형식으로 바운딩박스를 라벨 파일에 저장
        with open(label_file_path, "w") as label_file:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                # 바운딩박스를 YOLO 형식으로 정규화
                x_center_normalized = (x_min + x_max) / 2 / image_width
                y_center_normalized = (y_min + y_max) / 2 / image_height
                width_normalized = (x_max - x_min) / image_width
                height_normalized = (y_max - y_min) / image_height

                # 라벨 파일에 바운딩박스 정보 작성
                label_file.write(f"{class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n")

print("✅ 20개의 바운딩박스가 추가된 이미지를 저장하고 라벨 파일 생성 완료!")
