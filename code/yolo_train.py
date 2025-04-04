import open3d as o3d
import numpy as np

# .xyz 파일 경로
file_path = '/Users/jiwan/Desktop/yolo_fl/source/14 Ladybrook Road 10.ply'

# 저장할 .ply 파일 경로
output_file_path = '/Users/jiwan/Desktop/yolo_fl/source'

# .xyz 파일 읽기
pcd = o3d.io.read_point_cloud(file_path)

# 포인트 좌표 가져오기
points = np.asarray(pcd.points)

# z 좌표를 기준으로 색상 변환
z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

# 색상을 깊이에 따라 선형적으로 변환 (예: 빨간색 → 파란색)
colors = np.zeros((len(points), 3))  # RGB 색상 배열 (초기화)

for i, point in enumerate(points):
    # z 값에 비례해서 색상 변경 (선형적으로 변환)
    normalized_z = (point[2] - z_min) / (z_max - z_min)  # z 값을 0~1로 정규화
    # 빨간색 (normalized_z=0)에서 파란색 (normalized_z=1)으로 선형 변환
    colors[i] = [1 - normalized_z, 0, normalized_z]  # RGB 값 (빨간색에서 파란색으로 변화)

# 색상 정보 추가
pcd.colors = o3d.utility.Vector3dVector(colors)

# 포인트 클라우드를 파일로 저장
o3d.io.write_point_cloud(output_file_path, pcd)
print(f"Point cloud saved as '{output_file_path}'")

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd])

# 포인트 클라우드 점 정보 출력
print("Points (좌표 정보):\n", points)

# 색상 정보 출력
colors = np.asarray(pcd.colors)
print("Colors (색상 정보):\n", colors)

# 포인트 클라우드의 포인트 개수 출력
print(f"Number of points in the point cloud: {len(points)}")
