import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from eolearn.core import EOTask, EOPatch
from scipy.ndimage import distance_transform_edt, gaussian_filter, generic_filter
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# 1. 全局配置 (请在此处检查路径)
# --------------------------------------------------------------------------
CONFIG = {
    "IMAGE_PATH": r"C:\Users\wzc\Desktop\20220228 北师大珠海校区(2)\高光谱反射率数据\raw_10096_rd_rf_or2.tif",# 指向原有的高光谱数据文件
    "ENVI_NDVI_PATH": r"C:\Users\wzc\Desktop\测试2",  # 指向ENVI导出的数据文件
    "OUT_DIR": "Fire_Risk_Final_Project",
    "SWIR_BAND": 183,  # 用于建筑物识别的短波红外波段
    "NIR_BAND": 265,  # 用于植被/建筑区分的近红外波段
    "CNN_EPOCHS": 25,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)


# --------------------------------------------------------------------------
# 2. 任务定义
# --------------------------------------------------------------------------

class LoadDataTask(EOTask):
    """加载原始高光谱与ENVI特征"""

    def execute(self, eopatch=None):
        if eopatch is None: eopatch = EOPatch()

        # 加载原始HSI
        with rasterio.open(CONFIG['IMAGE_PATH']) as src:
            img = src.read().astype(np.float32).transpose(1, 2, 0)
            eopatch.data['HSI'] = np.nan_to_num(img)[np.newaxis, ...]
            eopatch.meta_info['profile'] = src.profile

        # 加载ENVI NDVI
        with rasterio.open(CONFIG['ENVI_NDVI_PATH']) as src:
            ndvi = src.read(1).astype(np.float32)
            eopatch.data['ENVI_NDVI'] = np.clip(ndvi, -1, 1)[np.newaxis, ..., np.newaxis]

        return eopatch


class FeatureEngineeringTask(EOTask):
    """计算建筑指数NDBI与增强型水体距离场"""

    def execute(self, eopatch):
        img = eopatch.data['HSI'][0]
        ndvi = eopatch.data['ENVI_NDVI'][0, ..., 0]

        # 1. 计算 NDBI
        swir = img[..., CONFIG['SWIR_BAND']]
        nir = img[..., CONFIG['NIR_BAND']]
        ndbi = (swir - nir) / (swir + nir + 1e-8)
        eopatch.data['NDBI'] = ndbi[np.newaxis, ..., np.newaxis]

        # 2. 增强型水体距离计算
        print("[*] 正在计算地理距离场...")
        # 改进：如果 NDVI 全大于 0，则取 NDVI 最低的 5% 作为疑似水体，防止距离场失效
        if np.min(ndvi) >= 0:
            water_thresh = np.percentile(ndvi, 5)
            water_mask = (ndvi <= water_thresh).astype(np.uint8)
        else:
            water_mask = (ndvi < 0).astype(np.uint8)

        dist_field = distance_transform_edt(1 - water_mask)

        # 3. 风险增益重构：使用对数拉伸，增强远距离的权重感
        # 让距离增加带来的风险上升更加明显
        dist_norm = dist_field / (np.max(dist_field) + 1e-8)
        dist_gain = np.power(dist_norm, 0.5)  # 开方处理，让中远距离的增益快速饱和

        eopatch.data['DIST_GAIN'] = dist_gain[np.newaxis, ..., np.newaxis]
        eopatch.mask_timeless['WATER_MASK'] = water_mask[..., np.newaxis]
        return eopatch


class HybridCNNTask(EOTask):
    """CNN分类与建筑抑制逻辑"""

    class Simple1DCNN(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                nn.Linear(16 * 8, 32), nn.ReLU(),
                nn.Linear(32, 1), nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x.unsqueeze(1))

    def execute(self, eopatch):
        img = eopatch.data['HSI'][0]
        ndvi = eopatch.data['ENVI_NDVI'][0, ..., 0]
        ndbi = eopatch.data['NDBI'][0, ..., 0]
        H, W, B = img.shape

        # 自动采样：利用ENVI NDVI作为高置信度标签
        flat_img = img.reshape(-1, B)
        flat_ndvi = ndvi.ravel()
        pos_idx = np.where(flat_ndvi > 0.6)[0]
        neg_idx = np.where(flat_ndvi < 0.1)[0]

        n = min(len(pos_idx), len(neg_idx), 4000)
        train_idx = np.concatenate([np.random.choice(pos_idx, n), np.random.choice(neg_idx, n)])

        X = torch.FloatTensor(flat_img[train_idx]).to(CONFIG['DEVICE'])
        y = torch.FloatTensor(np.concatenate([np.ones(n), np.zeros(n)])).unsqueeze(1).to(CONFIG['DEVICE'])

        model = self.Simple1DCNN(B).to(CONFIG['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(CONFIG['CNN_EPOCHS']):
            optimizer.zero_grad()
            loss = nn.BCELoss()(model(X), y)
            loss.backward();
            optimizer.step()

        # 推理
        model.eval()
        probs = np.zeros(H * W)
        with torch.no_grad():
            for i in range(0, H * W, 50000):
                end = min(i + 50000, H * W)
                probs[i:end] = model(torch.FloatTensor(flat_img[i:end]).to(CONFIG['DEVICE'])).cpu().numpy().flatten()

        # --- 核心改进：抑制建筑物 ---
        # 识别特征：NDBI高 且 NDVI不够高 的像素
        building_suppression = (ndbi.ravel() > 0.05) & (flat_ndvi < 0.45)
        probs[building_suppression] *= 0.1
        probs[flat_ndvi < 0.3] = 0  # 强制剔除裸地

        eopatch.data['PROB'] = probs.reshape(H, W)[np.newaxis, ..., np.newaxis]
        return eopatch


class FinalRiskAssessmentTask(EOTask):
    """重构火险融合算法：三因子加权法"""

    def execute(self, eopatch):
        prob = eopatch.data['PROB'][0, ..., 0]
        dist_gain = eopatch.data['DIST_GAIN'][0, ..., 0]
        ndvi = eopatch.data['ENVI_NDVI'][0, ..., 0]

        print("[*] 正在执行非线性风险融合...")

        # --- 核心算法改进 ---
        # 1. 植被密度因子 (树木概率的高次方，压制杂草，突出森林)
        vegetation_factor = np.power(prob, 1.5)

        # 2. 距离危险因子 (将 0-1 的增益映射到 1.0 - 3.0 倍率)
        # 离水越远，风险系数直接翻倍
        danger_multiplier = 1.0 + (dist_gain * 2.0)

        # 3. 综合风险
        raw_risk = vegetation_factor * danger_multiplier

        # 4. 动态范围拉伸 (98百分位拉伸，解决“不显红”问题)
        v_min, v_max = np.percentile(raw_risk, [2, 98])
        risk_final = np.clip((raw_risk - v_min) / (v_max - v_min + 1e-8), 0, 1)

        # 5. 水体绝对屏蔽
        risk_final[ndvi < 0.02] = 0

        # --- 绘图修正 (增加 vmin/vmax 确保 Risk Gain 显示) ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 树木概率图
        im0 = axes[0].imshow(prob, cmap='Greens', vmin=0, vmax=1)
        axes[0].set_title("1. Refined Tree Probability")
        plt.colorbar(im0, ax=axes[0])

        # 距离增益图 (显示为热力图，黄色代表离水远/高增益)
        im1 = axes[1].imshow(dist_gain, cmap='magma', vmin=0, vmax=1)
        axes[1].set_title("2. Normalized Distance Gain")
        plt.colorbar(im1, ax=axes[1])

        # 最终火险图
        im2 = axes[2].imshow(risk_final, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title("3. Final Fire Risk (Optimized)")
        plt.colorbar(im2, ax=axes[2])

        plt.savefig(os.path.join(CONFIG['OUT_DIR'], "Enhanced_Risk_Report.png"), dpi=300)

        # 导出 TIF
        profile = eopatch.meta_info['profile']
        profile.update(count=1, dtype=np.float32)
        with rasterio.open(os.path.join(CONFIG['OUT_DIR'], "Enhanced_Fire_Risk.tif"), 'w', **profile) as dst:
            dst.write(risk_final.astype(np.float32), 1)

        return eopatch


# --------------------------------------------------------------------------
# 3. 执行流程
# --------------------------------------------------------------------------
def run_all():
    print(f"[*] 启动高光谱火险评估流水线 (Device: {CONFIG['DEVICE']})")
    patch = EOPatch()

    tasks = [
        LoadDataTask(),
        FeatureEngineeringTask(),
        HybridCNNTask(),
        FinalRiskAssessmentTask()
    ]

    for task in tasks:
        print(f"    正在执行: {task.__class__.__name__}...")
        patch = task.execute(patch)

    print(f"\n[√] 全部流程完成！结果文件已生成至: {CONFIG['OUT_DIR']}")
    print("    - Final_Analysis_Report.png (多维对比报告图)")
    print("    - Fire_Risk_Map.tif (地理参考火险等级图)")


if __name__ == "__main__":

    run_all()
