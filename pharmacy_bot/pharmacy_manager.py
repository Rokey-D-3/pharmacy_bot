#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

from pharmacy_bot.srv import GetMedicineName, PickupMedicine, SrvDepthPosition

AVAILABLE_DRUGS = [
    "모드콜", "콜대원", "하이펜", "타이레놀", "다제스",
    "락토프린", "포비돈", "미니온", "퓨어밴드", "Rohto C3 Cube"
]

class PharmacyManager(Node):
    def __init__(self):
        super().__init__('pharmacy_manager')

        # 증상 또는 약 이름 수신
        self.subscription = self.create_subscription(
            String,
            '/symptom_text',
            self.symptom_callback,
            10
        )

        # 필요한 서비스 클라이언트 생성
        self.get_medicine_client = self.create_client(GetMedicineName, '/get_medicine_name')
        self.detect_position_client = self.create_client(SrvDepthPosition, '/get_3d_position')
        self.pickup_client = self.create_client(PickupMedicine, '/pickup_medicine')

        self.get_logger().info("PharmacyManager 실행됨 — 사용자 입력 대기 중")

    def symptom_callback(self, msg: String):
        user_input = msg.data.strip()

        # 1. 약 이름이면 바로 집기
        if user_input in AVAILABLE_DRUGS:
            self.get_logger().info(f"약 이름 직접 언급됨: {user_input}")
            self.process_medicine(user_input)
            return

        # 2. '__DONE__' 신호가 오면 symptom_query.txt 읽고 약 추천 수행
        if user_input == "__DONE__":
            try:
                path = os.path.expanduser("~/ros2_ws/src/pharmacy_bot/resource/symptom_query.txt")
                with open(path, "r", encoding="utf-8") as f:
                    symptom_text = f.read().strip()
                self.get_logger().info(f"최종 증상 텍스트: {symptom_text}")
            except Exception as e:
                self.get_logger().error(f"symptom_query.txt 로딩 실패: {e}")
                return

            recommended = self.call_get_medicine_name(symptom_text)
            if not recommended:
                self.get_logger().error("약 추천 실패")
                return
            self.get_logger().info(f"추천된 약: {recommended}")
            self.process_medicine(recommended)
            return

        # 그 외에는 단순 텍스트 로그만
        self.get_logger().info(f"(참고용) 입력 수신: \"{user_input}\"")

    def process_medicine(self, medicine_name: str):
        result = self.call_detect_position(medicine_name)
        if not result:
            self.get_logger().error("약 위치 탐지 실패")
            return

        position, width = result
        point = Point()
        point.x, point.y, point.z = position
        point.orientation.w = 1.0

        self.get_logger().info(f"{medicine_name}의 폭: {width * 1000:.1f} mm")
        
        success = self.call_pickup(point, width * 1000)
        if success:
            self.get_logger().info("약 집기 성공")
        else:
            self.get_logger().error("약 집기 실패")

    def call_get_medicine_name(self, symptom: str) -> str:
        if not self.get_medicine_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/get_medicine_name 서비스 연결 실패")
            return None

        request = GetMedicineName.Request()
        request.symptom = symptom
        future = self.get_medicine_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().medicine if future.result() else None

    def call_detect_position(self, medicine: str):
        if not self.detect_position_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/get_3d_position 서비스 연결 실패")
            return None

        request = SrvDepthPosition.Request()
        request.target = medicine
        future = self.detect_position_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if not result or sum(result.depth_position) == 0.0:
            return None
        return result.depth_position, result.width

    def call_pickup(self, point: Point, width: float) -> bool:
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/pickup_medicine 서비스 연결 실패")
            return False

        request = PickupMedicine.Request()
        request.point = point
        request.width = width
        future = self.pickup_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().success if future.result() else False


def main(args=None):
    rclpy.init(args=args)
    node = PharmacyManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

