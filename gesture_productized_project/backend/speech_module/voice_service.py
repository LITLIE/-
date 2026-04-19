from __future__ import annotations

import dashscope
from dashscope.audio.asr import Recognition
import base64
import os
import tempfile
import threading
from typing import Optional

import numpy as np

from config.settings import Config
from config.utils import Utils

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False
    sd = None

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except Exception:
    SPEECH_RECOGNITION_AVAILABLE = False
    sr = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    pyttsx3 = None


class VoiceService:
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.asr_enabled = False

        if OPENAI_AVAILABLE and config.asr_api_key:
            try:
                self.client = OpenAI(
                    api_key=config.asr_api_key,
                    base_url=config.asr_base_url,
                    timeout=45,
                )
                self.asr_enabled = True
                print(f'✅ ASR 客户端初始化成功，模型: {config.asr_model}')
            except Exception as e:
                print(f'❌ ASR 客户端初始化失败: {e}')
                self.client = None
                self.asr_enabled = False

        self.tts_engine = None
        if TTS_AVAILABLE and config.tts_enabled:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.9)
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    voice_id = getattr(voice, 'id', '') or ''
                    if 'zh' in voice_id.lower() or 'chinese' in voice_id.lower():
                        self.tts_engine.setProperty('voice', voice_id)
                        break
                print('✅ TTS 引擎初始化成功')
            except Exception as e:
                print(f'❌ TTS 引擎初始化失败: {e}')
                self.tts_engine = None

        self.recording = False
        self.status_text = '语音: 就绪'
        self.latest_transcript = ''
        self.lock = threading.Lock()
        self._check_audio_device()

    def _check_audio_device(self):
        if not self.can_record():
            self.status_text = '语音: 无录音设备或 ASR 未就绪'
            return False

        try:
            if SOUNDDEVICE_AVAILABLE:
                devices = sd.query_devices()
                input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
                if not input_devices:
                    self.status_text = '语音: 无麦克风输入'
                    return False
                return True
            if SPEECH_RECOGNITION_AVAILABLE:
                with sr.Microphone() as _:
                    return True
        except Exception as e:
            self.status_text = f'语音: 设备异常 {str(e)[:30]}'
            return False

        return False

    def can_record(self) -> bool:
        return (SOUNDDEVICE_AVAILABLE or SPEECH_RECOGNITION_AVAILABLE) and self.asr_enabled

    @staticmethod
    def _audio_mime_type(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.wav', '.wave'):
            return 'audio/wav'
        if ext == '.mp3':
            return 'audio/mpeg'
        if ext in ('.m4a', '.mp4'):
            return 'audio/mp4'
        if ext == '.aac':
            return 'audio/aac'
        if ext == '.flac':
            return 'audio/flac'
        return 'audio/wav'

    @staticmethod
    def _detect_silence(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        rms = float(np.sqrt(np.mean(np.square(audio_data)))) if audio_data.size else 1.0
        return rms < threshold

    def transcribe_file(self, path: str) -> str:
        # 如果 OpenAI 客户端没有成功初始化，直接返回
        if not self.client:
            print('❌ ASR 客户端未初始化，无法调用接口')
            return ""

        try:
            # 1. 读取音频文件为字节流
            with open(path, "rb") as f:
                audio_bytes = f.read()

            # 2. 获取 mime_type 并拼接 data_uri
            mime_type = self._audio_mime_type(path)
            data_uri = f"data:{mime_type};base64,{base64.b64encode(audio_bytes).decode()}"

            # 3. 使用 OpenAI 兼容接口发起请求 (完美运行版里的核心逻辑)
            res = self.client.chat.completions.create(
                model=self.config.asr_model,  # 也就是 qwen3-asr-flash
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": data_uri},
                            }
                        ],
                    }
                ],
                stream=False,
                extra_body={"asr_options": {"enable_itn": False}},
            )

            # 4. 解析返回值
            transcript = (res.choices[0].message.content or "").strip()
            return transcript

        except Exception as e:
            print(f'❌ ASR 接口调用崩溃: {e}')
            return ""

    def start_record_and_transcribe_async(self, duration: float, sample_rate: int, on_done):
        with self.lock:
            if self.recording:
                return
            if not self.can_record():
                on_done('', '无可用录音设备或 ASR 未初始化')
                return
            self.recording = True

        self.status_text = f'语音: 录音中 {duration:.0f}s...'
        threading.Thread(
            target=self._record_and_transcribe_task,
            args=(duration, sample_rate, on_done),
            daemon=True,
        ).start()

    def _record_and_transcribe_task(self, duration: float, sample_rate: int, on_done):
        transcript = ''
        err = ''
        tmp_path = None

        try:
            if SOUNDDEVICE_AVAILABLE:
                audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()
                audio = np.squeeze(audio)
                if self._detect_silence(audio):
                    err = '录音过于安静，请重新录制。'
                else:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        tmp_path = f.name
                    Utils.write_wav(tmp_path, audio, sample_rate)
                    transcript = self.transcribe_file(tmp_path)

            elif SPEECH_RECOGNITION_AVAILABLE:
                r = sr.Recognizer()
                with sr.Microphone(sample_rate=sample_rate) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source, duration=duration)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    tmp_path = f.name
                wav_bytes = audio_data.get_wav_data(convert_rate=sample_rate, convert_width=2)
                with open(tmp_path, 'wb') as wf:
                    wf.write(wav_bytes)
                transcript = self.transcribe_file(tmp_path)
            else:
                err = '缺少 sounddevice / speech_recognition，无法录音。'

            if not transcript and not err:
                err = '未识别到语音内容。'
        except Exception as e:
            err = f'语音处理失败：{str(e)[:120]}'
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.latest_transcript = transcript
            self.status_text = '语音: 已完成' if transcript else f'语音: {err}'
            try:
                on_done(transcript, err)
            except Exception:
                pass
            with self.lock:
                self.recording = False

    def speak_async(self, text: str):
        if not self.tts_engine or not text.strip():
            return
        threading.Thread(target=self._speak_task, args=(text[:150].strip(),), daemon=True).start()

    def _speak_task(self, text: str):
        try:
            self.tts_engine.stop()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass
