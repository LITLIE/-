from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Tuple

from config.settings import Config, SCENES

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None


class AIService:
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        if OPENAI_AVAILABLE and config.llm_api_key:
            try:
                self.client = OpenAI(api_key=config.llm_api_key, base_url=config.llm_base_url)
            except Exception:
                self.client = None

        self.history: Deque[Tuple[str, str]] = deque(maxlen=10)
        self.latest_response = '等待触发...'
        self.latest_user_text = ''
        self.is_processing = False
        self.lock = threading.Lock()

    def ask_async(self, scene_key: str, gesture: str, transcript: str, on_done=None):
        if not self.client:
            self.latest_response = 'LLM 未初始化：请检查 LLM_API_KEY / LLM_BASE_URL。'
            if on_done:
                on_done(self.latest_response)
            return

        user_text = (transcript or '').strip()
        if not user_text:
            user_text = f'当前手势：{gesture}。请结合场景给出最合适的建议。'

        with self.lock:
            if self.is_processing:
                return
            self.is_processing = True

        self.latest_user_text = user_text
        self.latest_response = '思考中...'
        threading.Thread(target=self._task, args=(scene_key, gesture, user_text, on_done), daemon=True).start()

    def _task(self, scene_key: str, gesture: str, user_text: str, on_done=None):
        try:
            scene = SCENES.get(scene_key, SCENES.get(self.config.scene_default, SCENES['meeting']))
            system_prompt = scene['prompt'] + (
                '\n请严格结合用户语音、当前手势和场景状态做回应。'
                '输出尽量不超过 90 个汉字，直接给可执行结果，不要长篇解释。'
            )

            messages = [{'role': 'system', 'content': system_prompt}]
            for role, content in self.history:
                messages.append({'role': role, 'content': content})

            user_prompt = (
                f"场景：{scene['name']}\n"
                f"当前手势：{gesture}\n"
                f"语音内容：{user_text}\n"
                f"请给出回复。"
            )
            messages.append({'role': 'user', 'content': user_prompt})

            res = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=0.5,
            )
            content = (res.choices[0].message.content or '').strip()
            self.latest_response = content or '模型没有返回有效内容。'
            self.history.append(('user', user_prompt))
            self.history.append(('assistant', self.latest_response))
        except Exception as e:
            self.latest_response = f'LLM 请求出错：{str(e)[:140]}'
        finally:
            with self.lock:
                self.is_processing = False
            if on_done:
                on_done(self.latest_response)
