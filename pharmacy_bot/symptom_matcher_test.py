#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pharmacy_bot.srv import GetMedicineName  # 서비스 타입 임포트

class SymptomMatcherClient(Node):
    def __init__(self):
        super().__init__('symptom_matcher_client')
        self.cli = self.create_client(GetMedicineName, '/get_medicine_name')
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('서비스를 기다리는 중입니다...')

        self.req = GetMedicineName.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)
    client = SymptomMatcherClient()

    # 요청 전 symptom_query.txt 확인
    resource_path = "/home/choin/ros2_ws/src/pharmacy_bot/resource/symptom_query.txt"
    with open(resource_path, "w", encoding="utf-8") as f:
        f.write("열이 나고 두통이 있어요")  # 테스트용 증상

    client.send_request()

    while rclpy.ok():
        rclpy.spin_once(client)
        if client.future.done():
            try:
                response = client.future.result()
            except Exception as e:
                client.get_logger().error(f"서비스 호출 실패: {e}")
            else:
                client.get_logger().info(f"추천 약: {response.medicine}")
            break

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
