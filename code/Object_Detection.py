import torch
from pathlib import Path
import cv2
import numpy as np

# 모델 로드
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)  # 커스텀 모델 로드
    return model

# 이미지에서 객체 인식 수행
def detect_objects(model, image_path, conf_thres=0.25, iou_thres=0.45):
    # 이미지 로드
    img = cv2.imread(image_path)
    
    # 이미지를 YOLO 모델의 입력 크기에 맞게 resize
    img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    img_resized = cv2.resize(img_resized, (640, 640))  # 모델이 요구하는 크기 (640x640)

    # 객체 탐지
    results = model(img_resized)
    
    # 결과 출력
    results.show()  # 이미지를 화면에 띄움
    
    # 결과 정보 가져오기
    labels, _, _ = results.xywh[0], results.pandas().xywh, results.names  # 인식된 객체의 라벨
    return labels, results.pandas().xywh

# 이미지 경로 설정
image_path = "/Users/jiwan/Desktop/yolo_fl/clustering/full_cloud_image_xy.png"  # 인식할 이미지 파일 경로
weights_path = "/Users/jiwan/Desktop/yolo_fl/yolov5/runs/train/sofa_custom4/weights/best.pt"

# 모델 로드
model = load_model(weights_path)

# 객체 인식
labels, detected_objects = detect_objects(model, image_path)

# 인식된 객체의 라벨 출력
print("Detected objects:", labels)
