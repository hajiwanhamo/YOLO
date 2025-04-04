import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os

# 포인트 클라우드 파일 로드
pcd = o3d.io.read_point_cloud("/Users/jiwan/Desktop/yolo_fl/source/14 Ladybrook Road 10.ply")
points = np.asarray(pcd.points)

# 포인트 클라우드 샘플링
sample_rate = 0.08  # 8% 샘플링
num_points = points.shape[0]
sampled_points = points[np.random.choice(num_points, int(num_points * sample_rate), replace=False)]

# DBSCAN 클러스터링
db = DBSCAN(eps=0.1, min_samples=50).fit(sampled_points)
labels = db.labels_

# 클러스터 수 확인
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"클러스터 수: {num_clusters}")

# 클러스터별 속성 저장
cluster_data = []
for i in range(num_clusters):
    cluster_points = sampled_points[labels == i]
    
    # 클러스터 중심 좌표
    cluster_center = np.mean(cluster_points, axis=0)
    
    # 볼륨 & 면적 계산 (Convex Hull 사용)
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        volume = hull.volume
        area = hull.area
    else:
        volume, area = 0, 0

    # 클러스터 정보 저장
    cluster_data.append({
        "id": i,
        "center": cluster_center,
        "volume": volume,
        "area": area,
        "points": cluster_points
    })

# 자동 분류 로직: 소파와 테이블 구분
sofas = []
tables = []
for data in cluster_data:
    center_z = data["center"][2]  # 높이 값
    volume = data["volume"]
    area = data["area"]

    # 높이(z축)와 부피를 기준으로 분류
    if center_z > 0.5 and volume > 0.2:  # 높이가 높고 부피가 큰 경우 (소파로 판단)
        sofas.append(data)
    elif center_z < 0.5 and area > 0.3:  # 낮고 면적이 큰 경우 (테이블로 판단)
        tables.append(data)

# 결과 출력
print(f"감지된 소파 개수: {len(sofas)}")
print(f"감지된 테이블 개수: {len(tables)}")

# 색상 설정: 소파는 빨강, 테이블은 파랑
sofa_color = np.array([[1, 0, 0]] * len(sofas))  # 빨강

# 소파 포인트 클라우드 생성
sofa_pcd = o3d.geometry.PointCloud()
sofa_pcd.points = o3d.utility.Vector3dVector(np.vstack([s["points"] for s in sofas]))
sofa_pcd.colors = o3d.utility.Vector3dVector(np.vstack([sofa_color]))

# 테이블 포인트 클라우드 생성 (테이블이 없으면 빈 클라우드 생성)
if len(tables) > 0:
    table_color = np.array([[0, 0, 1]] * len(tables))  # 파랑
    table_pcd = o3d.geometry.PointCloud()
    table_pcd.points = o3d.utility.Vector3dVector(np.vstack([t["points"] for t in tables]))
    table_pcd.colors = o3d.utility.Vector3dVector(np.vstack([table_color]))
else:
    table_pcd = None

# 시각화
o3d.visualization.draw_geometries([sofa_pcd, table_pcd] if table_pcd else [sofa_pcd])

# 저장 경로 설정
output_dir = "/Users/jiwan/Desktop/yolo_fl/clustering"
os.makedirs(output_dir, exist_ok=True)

# 소파와 테이블을 파일로 저장
o3d.io.write_point_cloud(os.path.join(output_dir, "sofas.ply"), sofa_pcd)
if table_pcd:
    o3d.io.write_point_cloud(os.path.join(output_dir, "tables.ply"), table_pcd)

# 3D 포인트 클라우드 시각화 후 이미지로 저장
def save_3d_image(points, filename="3d_image.png"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 포인트를 scatter plot으로 시각화
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap=plt.cm.jet, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=60, azim=0)  # xy 평면을 위에서 바라보는 시점
    
    # 이미지 저장
    plt.savefig(filename, dpi=300)
    plt.close()

# 전체 포인트 클라우드를 이미지로 저장
save_3d_image(points, os.path.join(output_dir, "full_cloud_image_xy60.png"))
