import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 이미지가 저장된 디렉토리 경로 설정
image_folder = "/Users/jiwan/Desktop/yolo_fl/dataset/images/Train/"

# YOLO 모델 결과에 맞는 바운딩박스 좌표 (정답 설정)
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

# 이미지 폴더에서 이미지 파일을 가져오기
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

# 이미지와 라벨 파일 생성
for i, image_path in enumerate(image_paths):
    # 이미지 로드
    img = Image.open(image_path).convert('RGB')
    
    # 바운딩박스 좌표 추출
    bbox = bounding_boxes[i % len(bounding_boxes)]["coords"]
    x_min, y_min, x_max, y_max = bbox
    
    # 바운딩박스를 YOLO 형식으로 변환 (정규화)
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    
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

    # 이미지 보여주기
    plt.show()

print("✅ 바운딩박스가 추가된 이미지를 저장하고 라벨 파일 생성 완료!")
