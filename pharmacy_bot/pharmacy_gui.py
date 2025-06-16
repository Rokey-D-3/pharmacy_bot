#!/usr/bin/env python3

import sys
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget,
    QVBoxLayout, QTextEdit, QHBoxLayout
)
from PyQt6.QtCore import QTimer

class PharmacyGUI(Node, QWidget):
    def __init__(self):
        Node.__init__(self, 'pharmacy_gui')
        QWidget.__init__(self)

        # ───────────── GUI 레이아웃 설정 ─────────────
        self.setWindowTitle("Pharmacy Bot GUI")
        self.setGeometry(100, 100, 1000, 600)

        # 채팅 로그 창 (사용자 ↔ 로봇 대화)
        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setPlaceholderText("[user], [robot] 대화 로그 표시")

        # 약 추천 결과 창
        self.result_log = QTextEdit()
        self.result_log.setReadOnly(True)
        self.result_log.setPlaceholderText("약 추천 결과 및 복용 설명 표시")

        # 수직 박스 레이아웃
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("사용자와 로봇의 대화"))
        self.layout.addWidget(self.chat_log)
        self.layout.addWidget(QLabel("추천된 약 정보"))
        self.layout.addWidget(self.result_log)
        self.setLayout(self.layout)

        # ROS2 구독 설정
        self.create_subscription(String, '/symptom_text', self.on_symptom_text, 10)
        self.create_subscription(String, '/chat_log', self.on_chat_log, 10)
        self.create_subscription(String, '/recommended_drug', self.on_recommendation, 10)

    def on_symptom_text(self, msg):
        text = msg.data
        # "__DONE__" 제외하고 모두 대화창에 추가
        if text != "__DONE__":
            self.chat_log.append(text)

    def on_chat_log(self, msg):
        self.chat_log.append(msg.data)

    def on_recommendation(self, msg):
        # 전체 결과 텍스트를 그대로 추천 결과 창에 출력
        self.result_log.setPlainText(msg.data)


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    gui = PharmacyGUI()
    gui.show()

    # Qt GUI와 ROS2 spin 병렬 실행
    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(gui, timeout_sec=0.1))
    timer.start(50)

    sys.exit(app.exec())
    gui.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

