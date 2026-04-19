from __future__ import annotations

import copy
import itertools
import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIA_PIPE_AVAILABLE = hasattr(mp, 'solutions')
except Exception:
    mp = None
    MEDIA_PIPE_AVAILABLE = False

from config.utils import Utils
from .keypoint_classifier import KeyPointClassifier


@dataclass
class HandObservation:
    gesture: str
    confidence: float
    center: Tuple[int, int]
    handedness: str = 'Unknown'


@dataclass
class FrameObservation:
    hands: List[HandObservation]
    primary_gesture: str
    primary_confidence: float
    center: Tuple[int, int]
    motion_gesture: str = '无'
    hand_count: int = 0


class GestureEngine:
    # 与训练标签一一对应：请确保你训练时的类别顺序就是这个顺序
    MODEL_LABELS = ['Open', 'Close', 'Pointer', 'Victory', 'OK']
    MODEL_TO_CHINESE = {
        'Open': '五指张开',
        'Close': '握拳',
        'Pointer': '食指指向',
        'Victory': '胜利',
        'OK': 'OK',
    }

    def __init__(self, model_path: Optional[str] = None, num_threads: int = 1):
        if not MEDIA_PIPE_AVAILABLE:
            raise RuntimeError('mediapipe not installed. Please install mediapipe first.')

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend([
            os.path.join(current_dir, 'model', 'keypoint_classifier.tflite'),
            os.path.join(os.getcwd(), 'keypoint_classifier.tflite'),
            os.path.join(os.getcwd(), 'backend', 'gesture_module', 'model', 'keypoint_classifier.tflite'),
        ])
        resolved = None
        for p in candidates:
            if p and os.path.exists(p):
                resolved = p
                break
        if resolved is None:
            raise FileNotFoundError('无法加载模型，请确保 keypoint_classifier.tflite 在正确位置')

        self.model_path = resolved
        self.classifier = KeyPointClassifier(self.model_path, num_threads=num_threads)
        self.history: Deque[str] = deque(maxlen=7)
        self.motion_history: Deque[Tuple[float, float, float]] = deque(maxlen=12)
        self.index_tip_history: Deque[Tuple[int, int]] = deque(maxlen=12)
        self.available = True

    @staticmethod
    def _pt(lm, idx, w, h):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    @staticmethod
    def _dist_landmarks(lm, idx1, idx2, w, h):
        p1 = (lm[idx1].x * w, lm[idx1].y * h)
        p2 = (lm[idx2].x * w, lm[idx2].y * h)
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _finger_extended_y(lm, tip, pip, margin=0.015):
        return lm[tip].y < lm[pip].y - margin

    @staticmethod
    def _finger_folded_y(lm, tip, pip, margin=0.01):
        return lm[tip].y > lm[pip].y + margin

    @staticmethod
    def _thumb_extended(lm, handedness_label=None):
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        thumb_mcp = lm[2]
        index_mcp = lm[5]
        wrist = lm[0]

        d_tip_wrist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
        d_ip_wrist = math.hypot(thumb_ip.x - wrist.x, thumb_ip.y - wrist.y)
        outward = d_tip_wrist > d_ip_wrist + 0.02

        if handedness_label == 'Right':
            side_rule = thumb_tip.x > thumb_ip.x - 0.02
        elif handedness_label == 'Left':
            side_rule = thumb_tip.x < thumb_ip.x + 0.02
        else:
            side_rule = abs(thumb_tip.x - thumb_mcp.x) > 0.03

        thumb_up_like = thumb_tip.y < index_mcp.y - 0.015
        return outward and side_rule and thumb_up_like

    @staticmethod
    def _pre_process_landmark(landmark_list):
        temp = copy.deepcopy(landmark_list)
        base_x, base_y = temp[0][0], temp[0][1]
        for i, p in enumerate(temp):
            temp[i][0] = temp[i][0] - base_x
            temp[i][1] = temp[i][1] - base_y
        flat = list(itertools.chain.from_iterable(temp))
        max_value = max(map(abs, flat)) if flat else 1
        if max_value == 0:
            max_value = 1
        return [n / max_value for n in flat]

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def process(self, frame_rgb):
        return self.hands.process(frame_rgb)

    def _heuristic_label(self, lm, handedness_label: Optional[str], w: int, h: int) -> Tuple[str, float]:
        thumb_ext = self._thumb_extended(lm, handedness_label)
        index_ext = self._finger_extended_y(lm, 8, 6)
        middle_ext = self._finger_extended_y(lm, 12, 10)
        ring_ext = self._finger_extended_y(lm, 16, 14)
        pinky_ext = self._finger_extended_y(lm, 20, 18)

        extended_count = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])
        fist_count = sum([
            self._finger_folded_y(lm, 8, 6),
            self._finger_folded_y(lm, 12, 10),
            self._finger_folded_y(lm, 16, 14),
            self._finger_folded_y(lm, 20, 18),
        ])

        thumb_tip_index_tip = self._dist_landmarks(lm, 4, 8, w, h)
        wrist = self._pt(lm, 0, w, h)
        thumb_tip = self._pt(lm, 4, w, h)
        palm_size = max(40.0, self._dist_landmarks(lm, 0, 9, w, h))

        ok_score = 0.0
        if thumb_tip_index_tip < palm_size * 0.30:
            ok_score += 0.7
        if middle_ext and ring_ext and pinky_ext:
            ok_score += 0.3

        thumbs_up_score = 0.0
        if thumb_ext:
            thumbs_up_score += 0.5
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            thumbs_up_score += 0.35
        if thumb_tip[1] < wrist[1]:
            thumbs_up_score += 0.15

        victory_score = 1.0 if index_ext and middle_ext and (not ring_ext) and (not pinky_ext) else 0.0
        one_score = 1.0 if index_ext and not middle_ext and not ring_ext and not pinky_ext else 0.0
        three_score = 1.0 if index_ext and middle_ext and ring_ext and (not pinky_ext) else 0.0
        four_score = 1.0 if index_ext and middle_ext and ring_ext and pinky_ext and (not thumb_ext) else 0.0
        open_score = 1.0 if index_ext and middle_ext and ring_ext and pinky_ext and thumb_ext else 0.0
        fist_score = 1.0 if fist_count >= 4 and extended_count <= 1 else 0.0

        candidates = [
            ('OK', ok_score),
            ('点赞', thumbs_up_score),
            ('胜利', victory_score),
            ('三', three_score),
            ('四', four_score),
            ('五指张开', open_score),
            ('握拳', fist_score),
            ('食指指向', one_score),
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = candidates[0]
        if best_score < 0.56:
            return '无', 0.0
        return best_label, float(best_score)

    def _classify_model(self, lm, w: int, h: int) -> Tuple[str, float]:
        landmark_list = [[int(p.x * w), int(p.y * h)] for p in lm]
        processed = self._pre_process_landmark(landmark_list)
        class_id, confidence, _ = self.classifier.predict(processed)
        raw_label = self.MODEL_LABELS[class_id] if 0 <= class_id < len(self.MODEL_LABELS) else '无'
        label = self.MODEL_TO_CHINESE.get(raw_label, raw_label)
        return label, confidence

    def classify_single_hand(self, hand_landmarks, handedness_label: Optional[str], w: int, h: int) -> Tuple[str, float]:
        lm = hand_landmarks.landmark
        model_label, model_conf = self._classify_model(lm, w, h)
        heuristic_label, heuristic_conf = self._heuristic_label(lm, handedness_label, w, h)

        if model_label != '无' and model_conf >= 0.42:
            if heuristic_label == model_label:
                return model_label, float(max(model_conf, heuristic_conf))
            if model_conf >= heuristic_conf:
                return model_label, float(model_conf)

        if heuristic_label != '无':
            return heuristic_label, float(heuristic_conf)

        if model_label != '无' and model_conf >= 0.30:
            return model_label, float(model_conf)

        return '无', 0.0

    def classify_two_hands(self, hand1, hand2, w: int, h: int) -> Tuple[str, float, Tuple[int, int]]:
        lm1 = hand1.landmark
        lm2 = hand2.landmark

        c1 = Utils.mid(self._pt(lm1, 0, w, h), self._pt(lm1, 9, w, h))
        c2 = Utils.mid(self._pt(lm2, 0, w, h), self._pt(lm2, 9, w, h))
        center = Utils.mid(c1, c2)
        center_dist = Utils.dist2d(c1, c2)

        thumb1 = self._pt(lm1, 4, w, h)
        thumb2 = self._pt(lm2, 4, w, h)
        index1 = self._pt(lm1, 8, w, h)
        index2 = self._pt(lm2, 8, w, h)

        thumb_dist = Utils.dist2d(thumb1, thumb2)
        index_dist = Utils.dist2d(index1, index2)

        p1_label, p1_score = self.classify_single_hand(hand1, None, w, h)
        p2_label, p2_score = self.classify_single_hand(hand2, None, w, h)

        if center_dist < w * 0.28 and thumb_dist < w * 0.12 and index_dist < w * 0.13:
            return '比心', 0.98, center
        if center_dist < w * 0.20 and thumb_dist < w * 0.18:
            return '双手合拢', 0.80, center
        if p1_label == '五指张开' and p2_label == '五指张开':
            return '双手张开', 0.86, center

        if p1_score >= p2_score:
            return p1_label, p1_score, c1
        return p2_label, p2_score, c2

    def detect_swipe(self, center_xy: Tuple[int, int]) -> str:
        t = time.time()
        self.motion_history.append((t, float(center_xy[0]), float(center_xy[1])))
        if len(self.motion_history) < self.motion_history.maxlen:
            return '无'

        t0, x0, y0 = self.motion_history[0]
        t1, x1, y1 = self.motion_history[-1]
        dt = t1 - t0
        if dt < 0.35:
            return '无'
        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) > 170 and abs(dy) < 120:
            self.motion_history.clear()
            return '右滑' if dx > 0 else '左滑'
        return '无'

    def identify(self, frame: np.ndarray) -> FrameObservation:
        if not self.available:
            return FrameObservation([], '无', 0.0, (0, 0))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        hands_info: List[HandObservation] = []
        raw_primary_gesture, primary_conf, primary_center = '无', 0.0, (0, 0)
        motion_gesture = '无'

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                h, w, _ = frame.shape
                handedness_label = None
                if idx < len(handedness_list):
                    try:
                        handedness_label = handedness_list[idx].classification[0].label
                    except Exception:
                        handedness_label = None

                label, conf = self.classify_single_hand(hand_landmarks, handedness_label, w, h)
                lm = hand_landmarks.landmark
                cx, cy = self._pt(lm, 9, w, h)
                hands_info.append(HandObservation(label, conf, (cx, cy), handedness_label or 'Unknown'))

                if conf > primary_conf:
                    primary_conf, raw_primary_gesture, primary_center = conf, label, (cx, cy)

                # 给滑动检测使用较稳的中心点
                self.index_tip_history.append(self._pt(lm, 8, w, h))
                self.draw_landmarks(frame, hand_landmarks)

            if len(results.multi_hand_landmarks) >= 2:
                g2, s2, c2 = self.classify_two_hands(
                    results.multi_hand_landmarks[0],
                    results.multi_hand_landmarks[1],
                    frame.shape[1],
                    frame.shape[0],
                )
                if s2 >= primary_conf:
                    raw_primary_gesture, primary_conf, primary_center = g2, s2, c2

            motion_gesture = self.detect_swipe(primary_center)

        else:
            self.index_tip_history.clear()
            self.motion_history.clear()

        self.history.append(raw_primary_gesture)
        most_common, count = Counter(self.history).most_common(1)[0]
        smoothed_gesture = most_common if count >= max(1, int(len(self.history) * 0.6)) else '无'
        if smoothed_gesture == '无' and raw_primary_gesture != '无' and primary_conf >= 0.35:
            smoothed_gesture = raw_primary_gesture

        return FrameObservation(
            hands=hands_info,
            primary_gesture=smoothed_gesture,
            primary_confidence=primary_conf,
            center=primary_center,
            motion_gesture=motion_gesture,
            hand_count=len(hands_info),
        )

    def analyze(self, frame):
        obs = self.identify(frame)
        return obs.primary_gesture, obs.primary_confidence, obs.center
