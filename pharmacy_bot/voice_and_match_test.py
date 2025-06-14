#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pharmacy_bot.srv import GetMedicineName


class IntegratedPharmacyTester(Node):
    def __init__(self):
        super().__init__('integrated_pharmacy_tester')

        # symptom_text 토픽 구독
        self.subscription = self.create_subscription(
            String,
            '/symptom_text',
            self.symptom_callback,
            10
        )

        # GetMedicineName 서비스 클라이언트 생성
        self.cli = self.create_client(GetMedicineName, 'get_medicine_name')

        # 서비스가 준비될 때까지 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('symptom_matcher 서비스 대기 중...')

        self.get_logger().info("통합 테스트 노드 시작됨 — voice_input + symptom_matcher 통합")

    def symptom_callback(self, msg):
        symptom_text = msg.data
        self.get_logger().info(f"받은 증상 텍스트: {symptom_text}")

        # 서비스 요청 생성 및 전송
        request = GetMedicineName.Request()
        request.symptom = symptom_text  # 수정됨

        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            medicine = future.result().medicine  # 수정됨
            self.get_logger().info(f"추천 약: {medicine}")
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
