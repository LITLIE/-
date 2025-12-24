import os
import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import generic_filter, binary_opening
# 直接导入 segment_anything 避免 samgeo 报错
from segment_anything import sam_model_registry, SamPredictor

# ======================
# 1. 路径与参数配置
# ======================
IMAGE_PATH = r"C:\Users\wzc\Desktop\20220228 北师大珠海校区(2)\高光谱反射率数据\raw_1200_rd_rf_or2.tif"
OUT_DIR = "samgeo_final_stable"
os.makedirs(OUT_DIR, exist_ok=True)

# 自动寻找模型路径 (如果报错找不到模型，请手动修改下面的路径)
CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "sam_vit_h_4b8939.pth")
if not os.path.exists(CHECKPOINT_PATH):
    # 备用：尝试在当前目录找
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

# ======================
# 2. 读取高光谱影像
# ======================
with rasterio.open(IMAGE_PATH) as src:
    img = src.read().astype(np.float32)
    profile = src.profile # 获取原始元数据
    img = np.nan_to_num(img)

bands, H, W = img.shape
print(f"Image loaded: {bands} bands, {H}x{W} pixels")

# ======================
# 3. 特征计算 (NDVI & K-Means)
# ======================
# 计算 NDVI
band_means = img.reshape(bands, -1).mean(axis=1)
nir_idx, red_idx = np.argmax(band_means), np.argmin(band_means)
nir, red = img[nir_idx], img[red_idx]
ndvi = (nir - red) / (nir + red + 1e-6)
texture = generic_filter(ndvi, np.std, size=3)

# 准备 K-Means 数据
mask_veg = ndvi > 0.4
data_mask = (img.sum(axis=0) != 0)
veg_indices = np.where(mask_veg.flatten() & data_mask.flatten())[0]

# 执行聚类
print("Running K-Means...")
X_veg = img.reshape(bands, -1).T[veg_indices]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_veg = kmeans.fit_predict(X_veg)

# 映射回图像
cluster_map = np.full((H * W), -1, dtype=np.int32)
cluster_map[veg_indices] = labels_veg
cluster_map = cluster_map.reshape(H, W)

# 自动选树 (纹理最大的类)
class_textures = []
for i in range(3):
    if np.any(cluster_map == i):
        class_textures.append(texture[cluster_map == i].mean())
    else:
        class_textures.append(0)
tree_class = np.argmax(class_textures)
print(f"Identified Class {tree_class} as Trees")

# ======================
# 4. 初始化 SAM (官方方式)
# ======================
print(f"Loading SAM model from: {CHECKPOINT_PATH}...")
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Error: Model file not found at {CHECKPOINT_PATH}")
    print("Please download 'sam_vit_h_4b8939.pth' and place it in the project folder.")
    exit()

sam_model = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
sam_model.to(device=device)
predictor = SamPredictor(sam_model)

# ======================
# 5. 准备图像与提示点
# ======================
def stretch(d):
    p2, p98 = np.percentile(d, (2, 98))
    return np.clip((d - p2) / (p98 - p2 + 1e-6), 0, 1)

# 制作 RGB
rgb = np.stack([stretch(img[nir_idx]), stretch(img[red_idx]), stretch(img[(nir_idx+red_idx)//2])], axis=-1)
rgb_uint8 = (rgb * 255).astype(np.uint8)
rgb_uint8[~data_mask] = 0

print("Setting image for SAM...")
predictor.set_image(rgb_uint8)

# 生成正向点 (树)
tree_clean = binary_opening((cluster_map == tree_class), structure=np.ones((3,3)))
tree_ys, tree_xs = np.where(tree_clean)
if len(tree_xs) > 0:
    pos_idx = np.random.choice(len(tree_xs), min(len(tree_xs), 50), replace=False)
    pos_coords = np.column_stack([tree_xs[pos_idx], tree_ys[pos_idx]])
else:
    print("Warning: No tree pixels found!")
    pos_coords = np.empty((0, 2))

# 生成负向点 (背景)
bg_ys, bg_xs = np.where((cluster_map == -1) & data_mask)
if len(bg_xs) > 0:
    neg_idx = np.random.choice(len(bg_xs), min(len(bg_xs), 50), replace=False)
    neg_coords = np.column_stack([bg_xs[neg_idx], bg_ys[neg_idx]])
else:
    neg_coords = np.empty((0, 2))

input_points = np.concatenate([pos_coords, neg_coords], axis=0)
input_labels = np.array([1]*len(pos_coords) + [0]*len(neg_coords))

# ======================
# 6. 执行预测并保存 (已修复)
# ======================
print("Predicting...")
masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
final_mask = masks[0].astype(np.uint8)

# 【修复点在这里】：先更新 profile 字典，再打开文件
out_profile = profile.copy()
out_profile.update(count=1, dtype=rasterio.uint8, nodata=0)

save_path = os.path.join(OUT_DIR, "final_tree_mask.tif")
with rasterio.open(save_path, "w", **out_profile) as dst:
    dst.write(final_mask, 1)

print(f"Saved GeoTIFF to: {save_path}")

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(rgb_uint8); plt.title("RGB")
plt.subplot(132); plt.imshow(cluster_map);
if len(pos_coords) > 0: plt.scatter(pos_coords[:,0], pos_coords[:,1], c='r', s=1)
plt.title("Prompts")
plt.subplot(133); plt.imshow(final_mask, cmap='Greens'); plt.title("Result")
plt.savefig(os.path.join(OUT_DIR, "final_plot.png"))
print("Done!")