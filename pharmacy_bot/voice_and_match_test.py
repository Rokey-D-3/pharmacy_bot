#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pharmacy_bot.srv import GetMedicineName
import os

class IntegratedPharmacyTester(Node):
    def __init__(self):
        super().__init__('integrated_pharmacy_tester')

        self.subscription = self.create_subscription(
            String,
            '/symptom_text',
            self.symptom_callback,
            10
        )

        self.cli = self.create_client(GetMedicineName, '/get_medicine_name')
        self.recommend_pub = self.create_publisher(String, '/recommended_drug', 10)

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('symptom_matcher 서비스 대기 중...')

        self.get_logger().info("통합 테스트 노드 시작됨 — voice_input + symptom_matcher 연동")

    def symptom_callback(self, msg):
        data = msg.data.strip()

        # "__DONE__" 신호가 들어왔을 때만 symptom_query.txt 기반 추천 수행
        if data != "__DONE__":
            self.get_logger().info(f"(참고) 수신된 텍스트: {data}")
            return

        # symptom_query.txt 읽기
        try:
            path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/resource/symptom_query.txt")
            with open(path, "r", encoding="utf-8") as f:
                symptom_text = f.read().strip()
            self.get_logger().info(f"최종 증상 텍스트: {symptom_text}")
        except Exception as e:
            self.get_logger().error(f"symptom_query.txt 읽기 실패: {e}")
            return

        # 서비스 요청 생성
        request = GetMedicineName.Request()
        request.symptom = symptom_text

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            medicine = future.result().medicine
            self.get_logger().info(f"추천 약: {medicine}")
            self.recommend_pub.publish(String(data=medicine))
        else:
            self.get_logger().error("약 추천 서비스 호출 실패")


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedPharmacyTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

