#!/usr/bin/env python3

#
# 完整可运行的火险评估管线（高光谱 + 可选 DSM 整合）
# - 基于用户原始脚本，整合：SAM 分割 -> 段级特征 -> 弱监督 RF -> 像素 CNN -> 段像素融合 -> 风险评估
# - 可选加载 DSM（如 ZhuHai AW3D30*.TIF）用于提升建筑/树木区分精度
#
# 使用方法示例：
#     python fire_risk_hsi_with_optional_dsm_OK.py ^
#   --hsi "C:\Users\wzc\Desktop\20220228 北师大珠海校区(2)\高光谱反射率数据\raw_10096_rd_rf_or2.tif" ^
#   --outdir "C:\Users\wzc\Desktop\output" ^
#   --use_dsm ^
#   --dsm_dir "C:\Users\wzc\Desktop\20220228 北师大珠海校区(2)\DSM"
#
# 依赖（建议虚拟环境）:
#     pip install rasterio numpy matplotlib spectral scikit-learn torch torchvision eolearn samgeo scipy joblib
#
# 注意：SamGeo 需要能访问 ViT 模型并可能需要 GPU。若没有 GPU 程序会回退到 CPU。

import os
import argparse
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import generic_filter, median_filter, uniform_filter, median_filter
from scipy.ndimage import generic_filter as ndi_generic_filter
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn

# EO-learn / SPy / SamGeo
from eolearn.core import EOTask, EOPatch
import spectral
try:
    from samgeo import SamGeo
except Exception:
    SamGeo = None

# -----------------------
# 配置与参数
# -----------------------
DEFAULT_CONFIG = {
    "IMAGE_PATH": None,
    "OUT_DIR": "eo_spy_output_v2",
    "SWIR_BAND": 183,
    "RED_BAND": None,
    "NIR_BAND": None,
    "NDVI_THRESHOLD_OBJ": 0.35,
    "NDVI_FALLBACK_THRESHOLD": 0.45,
    "MIN_POS_SAMPLES": 800,
    "MAX_SAMPLES": 15000,
    "PCA_COMPONENTS": 32,
    "CNN_EPOCHS": 15,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "USE_DSM": False,
    "DSM_PATH": None
}


# -----------------------
# 任务定义
# -----------------------
class LoadHSITask(EOTask):
    def __init__(self, hsi_path, use_dsm=False, dsm_path=None):
        self.hsi_path = hsi_path
        self.use_dsm = use_dsm
        self.dsm_path = dsm_path

    def execute(self, eopatch=None):
        if eopatch is None: eopatch = EOPatch()
        print(f"[*] Loading HSI: {self.hsi_path}")
        with rasterio.open(self.hsi_path) as src:
            img = src.read().astype(np.float32)
            img = np.transpose(img, (1, 2, 0))
            img = np.nan_to_num(img)
            eopatch.data['HSI'] = img[np.newaxis, ...]
            eopatch.meta_info['profile'] = src.profile
            eopatch.meta_info['crs'] = src.crs
            eopatch.meta_info['transform'] = src.transform

        if self.use_dsm and self.dsm_path and os.path.exists(self.dsm_path):
            print(f"[*] Loading DSM: {self.dsm_path}")
            with rasterio.open(self.dsm_path) as ds:
                dsm = ds.read(1).astype(np.float32)
                dsm = np.nan_to_num(dsm)
                # 如果 DSM 与 HSI 大小不一致，进行简单重采样（最近邻）
                hsi_shape = eopatch.data['HSI'].shape[1:3]
                if dsm.shape != tuple(hsi_shape):
                    print("[!] DSM shape differs from HSI. Resampling DSM to HSI shape (nearest).")
                    # naive resample
                    dsm_resampled = np.zeros(hsi_shape, dtype=np.float32)
                    ys = np.linspace(0, dsm.shape[0]-1, hsi_shape[0]).astype(int)
                    xs = np.linspace(0, dsm.shape[1]-1, hsi_shape[1]).astype(int)
                    for i, yy in enumerate(ys):
                        for j, xx in enumerate(xs):
                            dsm_resampled[i, j] = dsm[yy, xx]
                    dsm = dsm_resampled
                eopatch.data['DSM'] = dsm[np.newaxis, ..., np.newaxis]
        return eopatch


class SpectralFeatureTask(EOTask):
    def execute(self, eopatch):
        img = eopatch.data['HSI'][0].astype(np.float32)
        H, W, B = img.shape
        means = np.mean(img, axis=(0, 1))
        nir = DEFAULT_CONFIG['NIR_BAND'] if DEFAULT_CONFIG['NIR_BAND'] is not None else int(np.argmax(means))
        red = DEFAULT_CONFIG['RED_BAND'] if DEFAULT_CONFIG['RED_BAND'] is not None else int(np.argmax(means[:max(1, B//3)]))
        green = int(B//6) if B >= 6 else 1
        swir = min(DEFAULT_CONFIG['SWIR_BAND'], B - 1)

        print(f"[*] Selected bands -> green: {green}, red: {red}, nir: {nir}, swir: {swir}")

        ndvi = (img[..., nir] - img[..., red]) / (img[..., nir] + img[..., red] + 1e-8)
        ndmi = (img[..., nir] - img[..., swir]) / (img[..., nir] + img[..., swir] + 1e-8)
        mndwi = (img[..., green] - img[..., swir]) / (img[..., green] + img[..., swir] + 1e-8)
        ndbi = (img[..., swir] - img[..., nir]) / (img[..., swir] + img[..., nir] + 1e-8)
        brightness = np.mean(img, axis=-1)

        eopatch.data['NDVI'] = np.clip(ndvi, -1, 1)[np.newaxis, ..., np.newaxis]
        eopatch.data['NDMI'] = np.clip(ndmi, -1, 1)[np.newaxis, ..., np.newaxis]
        eopatch.data['MNDWI'] = np.clip(mndwi, -1, 1)[np.newaxis, ..., np.newaxis]
        eopatch.data['NDBI'] = np.clip(ndbi, -1, 1)[np.newaxis, ..., np.newaxis]
        eopatch.data['BRIGHT'] = brightness[np.newaxis, ..., np.newaxis]

        try:
            rgb = spectral.get_rgb(img, [red, nir, green])
            rgb_uint8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            rgb_path = os.path.join(DEFAULT_CONFIG['OUT_DIR'], "sam_input_rgb.png")
            plt.imsave(rgb_path, rgb_uint8)
            eopatch.meta_info['rgb_path'] = rgb_path
        except Exception as e:
            print("[!] RGB save failed:", e)
            eopatch.meta_info['rgb_path'] = None

        eopatch.meta_info['bands_idx'] = dict(green=int(green), red=int(red), nir=int(nir), swir=int(swir))
        return eopatch


class SamGeoTask(EOTask):
    def execute(self, eopatch):
        print("[*] Running SAM segmentation...")
        if SamGeo is None:
            raise RuntimeError("SamGeo package not installed or failed to import. Install samgeo to continue.")

        sam = SamGeo(model_type="vit_h", device=DEFAULT_CONFIG['DEVICE'])
        ndvi = eopatch.data['NDVI'][0, ..., 0]
        veg_mask = ndvi > 0.4
        coords = np.argwhere(veg_mask)
        point_coords = None
        if len(coords) > 0:
            idx = np.random.choice(len(coords), min(400, len(coords)), replace=False)
            point_coords = coords[idx][:, [1, 0]].tolist()

        output_path = os.path.join(DEFAULT_CONFIG['OUT_DIR'], "sam_mask.tif")
        source = eopatch.meta_info.get('rgb_path', None)
        if source is None:
            # If rgb_path not available, try to generate temporary rgb
            tmp_rgb = os.path.join(DEFAULT_CONFIG['OUT_DIR'], 'tmp_rgb_for_sam.png')
            img = eopatch.data['HSI'][0]
            bands = eopatch.meta_info.get('bands_idx', None)
            if bands:
                r, g, b = bands['red'], bands['green'], bands['nir']
            else:
                r, g, b = 0, 1, 2
            try:
                rgb = spectral.get_rgb(img, [r, g, b])
                plt.imsave(tmp_rgb, (rgb*255).astype(np.uint8))
                source = tmp_rgb
            except Exception:
                raise RuntimeError("Cannot create RGB for SAM; provide a valid rgb in meta_info or install spectral.")

        sam.generate(source=source, point_coords=point_coords,
                     point_labels=[1]*len(point_coords) if point_coords else None, output=output_path)

        with rasterio.open(output_path) as s:
            segments = s.read(1)
        eopatch.mask_timeless['SEGMENTS'] = segments[..., np.newaxis]
        return eopatch


class SegmentFeatureTask(EOTask):
    def execute(self, eopatch):
        segments = eopatch.mask_timeless['SEGMENTS'][..., 0].astype(int)
        img = eopatch.data['HSI'][0].astype(np.float32)
        H, W, B = img.shape

        ndvi = eopatch.data['NDVI'][0, ..., 0]
        ndmi = eopatch.data['NDMI'][0, ..., 0]
        mndwi = eopatch.data['MNDWI'][0, ..., 0]
        ndbi = eopatch.data['NDBI'][0, ..., 0]
        bright = eopatch.data['BRIGHT'][0, ..., 0]

        ids = np.unique(segments)
        ids = ids[ids != 0]

        feat_list = []
        seg_ids = []
        local_std = generic_filter(ndvi, np.std, size=3)

        for uid in ids:
            mask = (segments == uid)
            if mask.sum() < 10:
                continue
            seg_ids.append(uid)
            band_means = img[mask].mean(axis=0)
            band_vars = img[mask].var(axis=0).mean()

            f_ndvi = ndvi[mask].mean()
            f_ndmi = ndmi[mask].mean()
            f_mndwi = mndwi[mask].mean()
            f_ndbi = ndbi[mask].mean()
            f_bright = bright[mask].mean()
            f_texture = local_std[mask].mean()
            f_area = mask.sum()

            band_slice = band_means[:min(8, B)]
            feat = np.concatenate([
                band_slice,
                [band_vars, f_ndvi, f_ndmi, f_mndwi, f_ndbi, f_bright, f_texture, f_area]
            ])
            feat_list.append(feat)

        if len(feat_list) == 0:
            eopatch.meta_info['segment_features'] = {'ids': np.array([]), 'features': np.zeros((0, 8 + 8))}
            return eopatch

        feats = np.stack(feat_list, axis=0)
        eopatch.meta_info['segment_features'] = {'ids': np.array(seg_ids), 'features': feats}
        return eopatch


class TreeLabelingTask(EOTask):
    def execute(self, eopatch):
        print("[*] Segment-based weak labeling + RF training...")
        seg_info = eopatch.meta_info.get('segment_features', None)
        if not seg_info or len(seg_info['ids']) == 0:
            print("[!] No segments/features; fallback to NDVI thresholding.")
            ndvi = eopatch.data['NDVI'][0, ..., 0]
            tree_mask = ndvi > DEFAULT_CONFIG['NDVI_THRESHOLD_OBJ']
            eopatch.mask_timeless['LABELS'] = tree_mask[..., np.newaxis].astype(np.uint8)
            return eopatch

        ids = seg_info['ids']
        feats = seg_info['features']

        labeled_idx = []
        labels = []
        for i, uid in enumerate(ids):
            ndvi_val = feats[i, -7]
            mndwi_val = feats[i, -5]
            ndbi_val = feats[i, -4]
            bright_val = feats[i, -3]
            area = feats[i, -1]

            if mndwi_val > 0.35 and area > 30:
                labeled_idx.append(i); labels.append(3)
            elif ndbi_val > 0.25 and bright_val > np.percentile(feats[:, -3], 60):
                labeled_idx.append(i); labels.append(2)
            elif ndvi_val > 0.45 and area > 50:
                labeled_idx.append(i); labels.append(1)
            else:
                continue

        if len(labeled_idx) < 10:
            print("[!] Too few auto-labeled segments; fallback to NDVI pixel thresholding.")
            ndvi = eopatch.data['NDVI'][0, ..., 0]
            tree_mask = ndvi > DEFAULT_CONFIG['NDVI_FALLBACK_THRESHOLD']
            eopatch.mask_timeless['LABELS'] = tree_mask[..., np.newaxis].astype(np.uint8)
            return eopatch

        X = feats[labeled_idx]
        y = np.array([1 if lab == 1 else 0 for lab in labels])

        rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        print("[*] RF trained on weak labels: pos/neg =", y.sum(), "/", len(y)-y.sum())

        seg_probs = rf.predict_proba(feats)[:, 1]

        segments = eopatch.mask_timeless['SEGMENTS'][..., 0].astype(int)
        seg_prob_map = np.zeros_like(segments, dtype=np.float32)
        id_to_idx = {int(uid): idx for idx, uid in enumerate(ids)}
        for uid in np.unique(segments):
            if uid == 0: continue
            if int(uid) in id_to_idx:
                seg_prob_map[segments == uid] = seg_probs[id_to_idx[int(uid)]]
            else:
                seg_prob_map[segments == uid] = 0.0

        pos_mask = seg_prob_map > 0.6
        if pos_mask.sum() < DEFAULT_CONFIG['MIN_POS_SAMPLES']:
            pos_mask |= (seg_prob_map > 0.45)

        eopatch.mask_timeless['LABELS'] = pos_mask[..., np.newaxis].astype(np.uint8)
        eopatch.mask_timeless['SEG_PROB'] = seg_prob_map[..., np.newaxis].astype(np.float32)
        eopatch.meta_info['rf_model'] = rf
        return eopatch


class CNNPredictTask(EOTask):
    def __init__(self, config):
        self.config = config
        self.device = config["DEVICE"]

    class Simple1DCNN(nn.Module):
        def __init__(self, in_channels, num_classes=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(32 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            return self.net(x)

    def execute(self, eopatch):
        img = eopatch.data['HSI'][0]
        ndvi = eopatch.data['NDVI'][0, ..., 0]
        labels = eopatch.mask_timeless['LABELS'][..., 0].ravel()
        H, W, B = img.shape

        print("[*] Preparing CNN training data...")
        flat_img = img.reshape(-1, B)
        flat_ndvi = ndvi.ravel()

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where((labels == 0) & (flat_ndvi < 0.2))[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            print("[!] Not enough pos/neg samples for CNN; skipping CNN and using seg_prob if available.")
            if 'SEG_PROB' in eopatch.mask_timeless:
                eopatch.data['PROB'] = eopatch.mask_timeless['SEG_PROB'][np.newaxis, ...]
                return eopatch
            else:
                # fallback simple NDVI
                flat_prob = (flat_ndvi > 0.5).astype(np.float32)
                eopatch.data['PROB'] = flat_prob.reshape(H, W)[np.newaxis, ..., np.newaxis]
                return eopatch

        n_samples = min(len(pos_idx), len(neg_idx), 5000)
        rng = np.random.RandomState(42)
        pos_choice = rng.choice(pos_idx, n_samples, replace=False)
        neg_choice = rng.choice(neg_idx, n_samples, replace=False)
        train_idx = np.concatenate([pos_choice, neg_choice])

        X_train = torch.FloatTensor(flat_img[train_idx]).to(self.device)
        y_train = torch.FloatTensor(np.concatenate([np.ones(n_samples), np.zeros(n_samples)])).unsqueeze(1).to(self.device)

        model = self.Simple1DCNN(in_channels=B).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        print(f"[*] Training CNN (epochs={self.config['CNN_EPOCHS']})...")
        model.train()
        for epoch in range(self.config['CNN_EPOCHS']):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{self.config['CNN_EPOCHS']} Loss={loss.item():.4f}")

        print("[*] Running CNN inference on full image (chunked)...")
        model.eval()
        probs = np.zeros(H * W, dtype=np.float32)
        chunk_size = 50000
        with torch.no_grad():
            for i in range(0, H * W, chunk_size):
                end = min(i + chunk_size, H * W)
                batch_x = torch.FloatTensor(flat_img[i:end]).to(self.device)
                probs[i:end] = model(batch_x).cpu().numpy().flatten()

        texture = generic_filter(ndvi, np.std, size=3).ravel()
        low_texture_mask = texture < np.percentile(texture, 30)
        high_ndvi_mask = flat_ndvi > 0.4
        correction_mask = np.logical_and(low_texture_mask, high_ndvi_mask)
        probs[correction_mask] *= 0.5

        seg_prob_map = None
        if 'SEG_PROB' in eopatch.mask_timeless:
            seg_prob_map = eopatch.mask_timeless['SEG_PROB'][..., 0].ravel()

        if seg_prob_map is not None and seg_prob_map.shape[0] == probs.shape[0]:
            alpha = 0.6
            probs = alpha * seg_prob_map + (1 - alpha) * probs

        eopatch.data['PROB'] = probs.reshape(H, W)[np.newaxis, ..., np.newaxis]
        return eopatch


class ExportTask(EOTask):
    def execute(self, eopatch):
        print("[*] Executing DSM-constrained fire risk reconstruction...")

        prob = eopatch.data['PROB'][0, ..., 0]
        ndmi = eopatch.data['NDMI'][0, ..., 0]
        mndwi = eopatch.data['MNDWI'][0, ..., 0]
        ndbi = eopatch.data['NDBI'][0, ..., 0]
        ndvi = eopatch.data['NDVI'][0, ..., 0]

        # ===============================
        # 1. 植被干燥度（NDMI → Dryness）
        # ===============================
        d_min, d_max = np.percentile(ndmi, [5, 95])
        dryness = 1.0 - np.clip((ndmi - d_min) / (d_max - d_min + 1e-8), 0, 1)

        # ===============================
        # 2. DSM 高度因子（核心修正）
        # ===============================
        if 'DSM' in eopatch.data and eopatch.data['DSM'] is not None:
            height = eopatch.data['DSM'][0, ..., 0]
            height = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)

            # 物理约束：低于 3m 不形成林火
            height_factor = np.clip((height - 3.0) / 15.0, 0, 1)
        else:
            print("[!] DSM not found, fallback to 2D risk")
            height_factor = np.ones_like(prob)

        # ===============================
        # 3. 地物强制抑制
        # ===============================
        water_mask = mndwi > 0.3
        nonveg_mask = ndvi < 0.15
        low_height_mask = height_factor < 0.1

        # ===============================
        # 4. 核心火险融合（门控）
        # ===============================
        risk = (prob ** 2) * dryness * height_factor
        risk[water_mask | nonveg_mask | low_height_mask] = 0.0

        # ===============================
        # 5. 动态拉伸
        # ===============================
        r_min, r_max = np.percentile(risk, [1, 99])
        risk_final = np.clip((risk - r_min) / (r_max - r_min + 1e-8), 0, 1)

        # ===============================
        # 6. 输出
        # ===============================
        profile = eopatch.meta_info['profile'].copy()
        profile.update(count=1, dtype=np.float32)

        out_path = os.path.join(DEFAULT_CONFIG['OUT_DIR'], "Optimized_Fire_Risk_DSM.tif")

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(risk_final.astype(np.float32), 1)

        plt.figure(figsize=(15, 5))
        plt.subplot(131);
        plt.imshow(prob, cmap='Greens');
        plt.title("Tree Probability")
        plt.subplot(132);
        plt.imshow(height_factor, cmap='terrain');
        plt.title("DSM Height Factor")
        plt.subplot(133);
        plt.imshow(risk_final, cmap='hot');
        plt.title("DSM-Constrained Fire Risk")
        plt.savefig(os.path.join(DEFAULT_CONFIG['OUT_DIR'], "DSM_Constrained_Result.png"))
        plt.close()

        return eopatch


# -----------------------
# 运行入口
# -----------------------

def run_pipeline(hsi_path, out_dir, use_dsm=False, dsm_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    DEFAULT_CONFIG['OUT_DIR'] = out_dir
    DEFAULT_CONFIG['IMAGE_PATH'] = hsi_path
    DEFAULT_CONFIG['USE_DSM'] = use_dsm
    if use_dsm and dsm_dir:
        # auto-detect AW3D30-like tif
        found = None
        for fname in os.listdir(dsm_dir):
            if fname.lower().endswith('.tif') and 'aw3d' in fname.lower():
                found = os.path.join(dsm_dir, fname); break
        if found is None:
            # fallback to any tif
            for fname in os.listdir(dsm_dir):
                if fname.lower().endswith('.tif'):
                    found = os.path.join(dsm_dir, fname); break
        DEFAULT_CONFIG['DSM_PATH'] = found

    patch = EOPatch()
    tasks = [
        LoadHSITask(hsi_path, use_dsm=use_dsm, dsm_path=DEFAULT_CONFIG.get('DSM_PATH')),
        SpectralFeatureTask(),
        SamGeoTask(),
        SegmentFeatureTask(),
        TreeLabelingTask(),
        CNNPredictTask(DEFAULT_CONFIG),
        ExportTask()
    ]

    for task in tasks:
        patch = task.execute(patch)

    print(f"\n[√] Pipeline finished. Outputs in: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hsi', dest='hsi', required=True, help='Path to HSI .tif')
    parser.add_argument('--outdir', dest='outdir', default='eo_spy_output_v2')
    parser.add_argument('--use_dsm', dest='use_dsm', action='store_true', help='Enable DSM integration')
    parser.add_argument('--dsm_dir', dest='dsm_dir', default=None, help='Directory containing DSM files like AW3D30.TIF')
    args = parser.parse_args()

    run_pipeline(args.hsi, args.outdir, use_dsm=args.use_dsm, dsm_dir=args.dsm_dir)
