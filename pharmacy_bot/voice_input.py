#!/usr/bin/env python3

import os
import time
import tempfile
import numpy as np
import pyaudio
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample
from dotenv import load_dotenv
import openai
from openwakeword.model import Model

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ───── 설정 상수 ─────
EXIT_KEYWORDS = ["없어요", "괜찮아요", "아니요", "종료", "그만"]

# ───── 마이크 설정 ─────
class MicConfig:
    chunk = 12000
    rate = 48000
    channels = 1
    fmt = pyaudio.paInt16
    buffer_size = 24000

class MicController:
    def __init__(self, config=MicConfig()):
        self.config = config
        self.audio = None
        self.stream = None

    def open_stream(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk
        )

    def close_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()


# ───── Wake word 감지 ─────
class WakeupWord:
    def __init__(self, buffer_size):
        model_path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/resource/hello_rokey_8332_32.tflite")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 없음: {model_path}")
        self.model = Model(wakeword_models=[model_path])
        self.model_name = os.path.basename(model_path).split(".")[0]
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print(f"wakeupword confidence: {confidence:.3f}")
        return confidence > 0.3


# ───── Whisper STT ─────
class STT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.duration = 10
        self.samplerate = 16000

    def speech2text(self):
        print("사용자 발화 대기 (10초)")
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav.write(tmp.name, self.samplerate, audio)
            try:
                with open(tmp.name, "rb") as f:
                    transcript = openai.Audio.transcribe("whisper-1", file=f, api_key=self.api_key)
                print("인식 결과:", transcript["text"])
                return transcript["text"]
            except Exception as e:
                print(f"STT 실패: {e}")
                return None


# ───── voice_input ROS 2 노드 ─────
class VoiceInputNode(Node):
    def __init__(self):
        super().__init__('voice_input')

        # 수정된 방식: 명시적 경로 지정
        env_path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/.env")
        load_dotenv(dotenv_path=env_path)
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            self.get_logger().error(f"OPENAI_API_KEY를 찾을 수 없습니다: {env_path}")
            return

        self.tts = pyttsx3.init()
        self.publisher = self.create_publisher(String, '/symptom_text', 10)
        self.get_logger().info("VoiceInputNode 실행됨")

        self.symptom_list = []
        self.start_listening()

    def speak(self, text: str):
        print(f"[robot]: {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    def start_listening(self):
        mic = MicController()
        mic.open_stream()
        wakeup = WakeupWord(mic.config.buffer_size)
        wakeup.set_stream(mic.stream)
        stt = STT(self.api_key)

        while rclpy.ok():
            if wakeup.is_wakeup():
                self.speak("안녕하세요, 약 이름 또는 증상을 말씀해주세요.")
                while True:
                    user_input = stt.speech2text()
                    if not user_input:
                        self.speak("죄송해요. 다시 말씀해주시겠어요?")
                        continue

                    # 종료 조건 검사
                    if any(kw in user_input for kw in EXIT_KEYWORDS):
                        self.speak("알겠습니다. 대화를 종료할게요.")
                        break

                    self.symptom_list.append(user_input)
                    self.publisher.publish(String(data=user_input))
                    self.speak("다른 증상이나 필요한 약이 있으신가요?")

                mic.close_stream()
                self.save_symptom_query()

    def save_symptom_query(self):
        if not self.symptom_list:
            return
        query = " 그리고 ".join(self.symptom_list)
        resource_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resource", "symptom_query.txt"))
        with open(resource_path, "w", encoding="utf-8") as f:
            f.write(query)
        self.get_logger().info(f"증상 저장 완료 → {resource_path}")
        self.speak("지금까지 말씀하신 증상들을 저장했어요.")


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
