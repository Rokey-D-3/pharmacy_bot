#!/usr/bin/env python3

# ROS 2 기본 모듈
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 음성 인식 및 환경 설정 관련 모듈
import os
from dotenv import load_dotenv
import pyaudio
import io
import wave
import numpy as np
from openwakeword.model import Model
from scipy.signal import resample
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import openai


# ──────────────── MicController (마이크 제어 클래스) ────────────────
# 마이크 설정값을 담는 구조체 클래스
class MicConfig:
    chunk: int = 12000
    rate: int = 48000
    channels: int = 1
    record_seconds: int = 5
    fmt: int = pyaudio.paInt16
    buffer_size: int = 24000

# 마이크 스트림 열고/닫고 오디오 녹음 및 wav 데이터 반환
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

    def record_audio(self) -> bytes:
        """5초 동안 오디오 녹음 후 WAV 바이트 스트림 반환"""
        self.open_stream()
        frames = []
        for _ in range(0, int(self.config.rate / self.config.chunk * self.config.record_seconds)):
            data = self.stream.read(self.config.chunk)
            frames.append(data)
        self.close_stream()

        wav_io = io.BytesIO()
        wf = wave.open(wav_io, 'wb')
        wf.setnchannels(self.config.channels)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.config.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return wav_io.getvalue()


# ──────────────── WakeupWord (WakeupWord 감지) ────────────────
class WakeupWord:
    MODEL_NAME = "hello_rokey_8332_32.tflite"   # 사용자 정의 WakeupWord

    def __init__(self, buffer_size):
        from openwakeword import utils
        utils.download_models()
        self.model = Model(wakeword_models=[self.MODEL_NAME])
        self.model_name = self.MODEL_NAME.split(".", maxsplit=1)[0]
        self.buffer_size = buffer_size
        self.stream = None

    def set_stream(self, stream):
        self.stream = stream

    def is_wakeup(self):
        """
        마이크 입력에서 일정 길이 버퍼를 읽어와 WakeupWord 유무 판단
        0.3 이상 confidence 시 WakeupWord 감지 성공
        """
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print(f"confidence: {confidence:.3f}")
        return confidence > 0.3


# ──────────────── STT (Whisper API를 이용한 음성 → 텍스트 변환) ────────────────
class STT:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 5
        self.samplerate = 16000

    def speech2text(self):
        print("음성 녹음을 시작합니다. 5초 동안 말해주세요...")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("Whisper로 전송 중...")

        # 임시 WAV 파일로 저장 후 Whisper API에 전송
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)
            with open(temp_wav.name, "rb") as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1", file=f, api_key=self.openai_api_key
                )
        print("STT 결과:", transcript["text"])
        return transcript["text"]


# ──────────────── ROS2 Node ────────────────
class VoiceInputNode(Node):
    def __init__(self):
        super().__init__('voice_input')

        # .env에서 OpenAI API 키 불러오기
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            self.get_logger().error(".env에서 OPENAI_API_KEY를 찾을 수 없습니다.")
            return

        # 증상 텍스트 발행을 위한 퍼블리셔 설정
        self.publisher = self.create_publisher(String, '/symptom_text', 10)
        self.get_logger().info("Voice Input Node 실행됨 — WakeupWord 대기 중...")

        # WakeupWord 감지 및 음성 인식 루프 시작
        self.start_listening()

    def start_listening(self):
        # 마이크 초기화 및 WakeupWord 감지 세팅
        mic = MicController()
        mic.open_stream()

        wakeup = WakeupWord(mic.config.buffer_size)
        wakeup.set_stream(mic.stream)

        while rclpy.ok():
            if wakeup.is_wakeup():
                self.get_logger().info("WakeupWord 감지됨! 사용자 발화를 기다리는 중...")

                stt = STT(self.openai_api_key)
                user_input = stt.speech2text()

                if user_input:
                    # 텍스트를 /symptom_text로 publish
                    msg = String()
                    msg.data = user_input
                    self.publisher.publish(msg)
                    self.get_logger().info(f"증상 텍스트 퍼블리시됨: '{user_input}'")

                self.get_logger().info("WakeupWord 대기 중...")
            else:
                continue


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
