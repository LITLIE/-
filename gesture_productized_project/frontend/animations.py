from __future__ import annotations

import math
import random
from typing import List

import cv2
import numpy as np


class Animation:
    def update(self):
        raise NotImplementedError

    def draw(self, frame):
        raise NotImplementedError

    def is_dead(self):
        return True


class AnimationManager:
    def __init__(self, limit=180):
        self.animations: List[Animation] = []
        self.limit = limit

    def add(self, anim: Animation):
        if len(self.animations) < self.limit:
            self.animations.append(anim)

    def update_and_draw(self, frame):
        active = []
        for a in self.animations:
            a.update()
            a.draw(frame)
            if not a.is_dead():
                active.append(a)
        self.animations = active


class RingPulse(Animation):
    def __init__(self, x, y, color=(0, 220, 255), start_radius=10, life=34):
        self.x, self.y = int(x), int(y)
        self.color = color
        self.radius = start_radius
        self.life = life

    def update(self):
        self.radius += 6
        self.life -= 1

    def draw(self, frame):
        alpha = max(self.life / 34.0, 0.0)
        col = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (self.x, self.y), int(self.radius), col, 2, cv2.LINE_AA)

    def is_dead(self):
        return self.life <= 0


class SparkParticle(Animation):
    def __init__(self, x, y, color=(255, 255, 255)):
        self.x = float(x) + random.uniform(-6, 6)
        self.y = float(y) + random.uniform(-6, 6)
        ang = random.uniform(0, 2 * math.pi)
        sp = random.uniform(2.5, 8.5)
        self.vx = math.cos(ang) * sp
        self.vy = math.sin(ang) * sp - random.uniform(0.0, 1.4)
        self.life = random.randint(22, 42)
        self.color = color
        self.size = random.randint(2, 4)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.18
        self.vx *= 0.98
        self.life -= 1

    def draw(self, frame):
        alpha = max(self.life / 42.0, 0.0)
        col = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, col, -1, cv2.LINE_AA)

    def is_dead(self):
        return self.life <= 0


class HeartBurst(Animation):
    def __init__(self, x, y, life=70):
        self.x, self.y = int(x), int(y)
        self.life = life
        self.angle = 0.0

    def update(self):
        self.angle += 0.1
        self.life -= 1

    def draw(self, frame):
        scale = 34 + int(10 * math.sin(self.angle * 2))
        pts = []
        for t in np.linspace(0, 2 * math.pi, 100):
            xs = 16 * np.sin(t) ** 3
            ys = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
            x = int(self.x + xs * scale / 16)
            y = int(self.y - ys * scale / 16)
            pts.append((x, y))
        alpha = max(self.life / 70.0, 0.0)
        col = (int(255 * alpha), int(80 * alpha), int(180 * alpha))
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, col, 2, cv2.LINE_AA)

    def is_dead(self):
        return self.life <= 0
