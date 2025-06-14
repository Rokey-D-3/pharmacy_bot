#!/usr/bin/env python3

# ────────────── ROS 2 기본 모듈 ──────────────
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ────────────── 기본 및 음성 관련 라이브러리 ──────────────
import os
import time
import pyaudio
import numpy as np
import tempfile
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wav
import openai
from scipy.signal import resample
from dotenv import load_dotenv
from openwakeword.model import Model

# ────────────── 약 이름 및 종료 키워드 ──────────────
AVAILABLE_DRUGS = [
    "모드콜", "콜대원", "하이펜", "타이레놀", "다제스",
    "락토프린", "포비돈", "미니온", "퓨어밴드", "Rohto C3 Cube"
]
EXIT_KEYWORDS = ["없어요", "괜찮아요", "아니요", "종료", "그만"]

# ───────────────────────────────────────────────
# 마이크 제어 클래스
# ───────────────────────────────────────────────
class MicConfig:
    chunk: int = 12000
    rate: int = 48000
    channels: int = 1
    record_seconds: int = 5
    fmt: int = pyaudio.paInt16
    buffer_size: int = 24000

class MicController:
    def __init__(self, config: MicConfig = MicConfig()):
        self.config = config
        self.audio = None
        self.stream = None
        self.sample_width = None

    def open_stream(self):
        self.audio = pyaudio.PyAudio()
        self.sample_width = self.audio.get_sample_size(self.config.fmt)
        self.stream = self.audio.open(
            format=self.config.fmt,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk,
        )

    def close_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
            self.audio = None

# ───────────────────────────────────────────────
# WakeupWord 감지 클래스
# ───────────────────────────────────────────────
class WakeupWord:
    MODEL_NAME = "hello_rokey_8332_32.tflite"

    def __init__(self, buffer_size):
        from openwakeword import utils
        utils.download_models()

        # 절대경로로 사용자 모델 지정
        model_path = "/home/choin/ros2_ws/src/pharmacy_bot/resource/hello_rokey_8332_32.tflite"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"지정된 wake word 모델이 존재하지 않습니다: {model_path}")
        
        self.model = Model(wakeword_models=[model_path])
        self.model_name = self.MODEL_NAME.split(".", maxsplit=1)[0]
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print(f"confidence: {confidence:.3f}")
        return confidence > 0.3

# ───────────────────────────────────────────────
# Whisper STT 클래스
# ───────────────────────────────────────────────
class STT:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 5
        self.samplerate = 16000

    def speech2text(self):
        print("사용자 발화를 기다리는 중... 5초 안에 말씀해주세요.")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        print("Whisper API에 전송 중...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)
            with open(temp_wav.name, "rb") as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1", file=f, api_key=self.openai_api_key
                )
        print("STT 결과:", transcript["text"])
        return transcript["text"]

# ───────────────────────────────────────────────
# voice_input ROS 2 노드
# ───────────────────────────────────────────────
class VoiceInputNode(Node):
    def __init__(self):
        super().__init__('voice_input')

        load_dotenv(dotenv_path="/home/choin/ros2_ws/src/pharmacy_bot/.env")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            self.get_logger().error(".env에서 OPENAI_API_KEY를 찾을 수 없습니다.")
            return
        
        self.symptom_list = []
        self.tts_engine = pyttsx3.init()
        self.publisher = self.create_publisher(String, '/symptom_text', 10)

        self.get_logger().info("VoiceInputNode 시작됨 — WakeupWord 감지 대기 중")
        self.start_listening()

    def speak(self, text: str):
        print(f"[로봇]: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def start_listening(self):
        mic = MicController()
        mic.open_stream()
        wakeup = WakeupWord(mic.config.buffer_size)
        wakeup.set_stream(mic.stream)
        stt = STT(self.openai_api_key)

        while rclpy.ok():
            if wakeup.is_wakeup():
                self.get_logger().info("WakeupWord 감지됨!")
                self.speak("안녕하세요, 무엇을 도와드릴까요?")
                time.sleep(0.5)
                user_input = stt.speech2text()

                if not user_input:
                    self.speak("죄송해요. 잘 이해하지 못했어요.")
                    continue

                if any(exit_kw in user_input for exit_kw in EXIT_KEYWORDS):
                    self.speak("알겠습니다. 대화를 종료할게요.")
                    break

                matched_drug = None
                for drug in AVAILABLE_DRUGS:
                    if drug in user_input:
                        matched_drug = drug
                        break

                msg = String()
                if matched_drug:
                    self.speak(f"{matched_drug}를 꺼내드릴게요.")
                    msg.data = matched_drug
                else:
                    self.speak("말씀하신 증상을 기록했어요.")
                    self.symptom_list.append(user_input)
                    msg.data = user_input
                self.publisher.publish(msg)

        mic.close_stream()
        self.save_symptom_query()

    def save_symptom_query(self):
        if not self.symptom_list:
            return
        query = " 그리고 ".join(self.symptom_list)

        package_path = os.path.dirname(os.path.abspath(__file__))
        resource_path = os.path.join(package_path, "..", "resource", "symptom_query.txt")
        resource_path = os.path.abspath(resource_path)

        with open(resource_path, "w", encoding="utf-8") as f:
            f.write(query)
        self.get_logger().info(f"증상 저장 완료 → {resource_path}")
        self.speak("지금까지 말씀하신 증상들을 저장했어요.")

# ───────────────────────────────────────────────
# main(): ROS 2 노드 실행
# ───────────────────────────────────────────────
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
