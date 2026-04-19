from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None


class KeyPointClassifier:
    def __init__(self, model_path: str, num_threads: int = 1):
        if not TF_AVAILABLE:
            raise RuntimeError('TensorFlow Lite 未安装，无法加载模型')

        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        out_shape = self.output_details[0].get('shape', [1, -1])
        self.num_classes = int(out_shape[-1]) if len(out_shape) else -1
        print(f'🔥 模型加载成功！手势类别数：{self.num_classes}')

    def predict(self, landmark_list: Sequence[float]):
        x = np.asarray([landmark_list], dtype=np.float32)
        input_idx = self.input_details[0]['index']
        self.interpreter.set_tensor(input_idx, x)
        self.interpreter.invoke()

        output_idx = self.output_details[0]['index']
        probs = np.asarray(self.interpreter.get_tensor(output_idx)[0], dtype=np.float32)

        # 兼容 logits / 概率两种输出
        if np.max(probs) > 1.0 or np.min(probs) < 0.0:
            e_x = np.exp(probs - np.max(probs))
            probs = e_x / np.sum(e_x)
        else:
            total = float(np.sum(probs))
            if total > 0:
                probs = probs / total

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return class_id, confidence, probs
