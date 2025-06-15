#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

from pharmacy_bot.srv import GetMedicineName, PickupMedicine
from od_msg.srv import SrvDepthPosition

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
        self.get_logger().info(f"입력 수신: \"{user_input}\"")

        # 직접 약 이름 언급한 경우
        if user_input in AVAILABLE_DRUGS:
            self.get_logger().info(f"약 이름 직접 언급됨: {user_input}")
            self.process_medicine(user_input)
            return

        # 증상으로 간주 → 약 추천 요청
        recommended = self.call_get_medicine_name(user_input)
        if not recommended:
            self.get_logger().error("약 추천 실패")
            return

        self.get_logger().info(f"추천된 약: {recommended}")
        self.process_medicine(recommended)

    def process_medicine(self, medicine_name: str):
        position = self.call_detect_position(medicine_name)
        if not position:
            self.get_logger().error("약 위치 탐지 실패")
            return

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.w = 1.0

        success = self.call_pickup(pose)
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
        return result.depth_position

    def call_pickup(self, pose: Pose) -> bool:
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/pickup_medicine 서비스 연결 실패")
            return False

        request = PickupMedicine.Request()
        request.pose = pose
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

