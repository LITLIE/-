from __future__ import annotations

import math
import wave
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import ImageFont


class Utils:
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def dist2d(a, b):
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def mid(a, b):
        return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

    @staticmethod
    def majority_vote(items: Sequence[str]) -> Tuple[str, int]:
        if not items:
            return '无', 0
        c = Counter(items)
        return c.most_common(1)[0]

    @staticmethod
    def write_wav(path: str, audio: np.ndarray, sample_rate: int):
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def load_font(size: int = 24):
        candidates = [
            'simhei.ttf',
            'msyh.ttc',
            'msyh.ttf',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/msyh.ttf',
            '/System/Library/Fonts/PingFang.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        for path in candidates:
            try:
                if Path(path).exists() or path.startswith('/usr') or path.startswith('C:/'):
                    return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()
