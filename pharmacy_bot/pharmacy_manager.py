#!/usr/bin/env python3

# ROS 2 기본 모듈 및 메시지/서비스
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

# 사용자 정의 서비스
from pharmacy_bot.srv import GetMedicineName, PickupMedicine  # 증상 → 약, 위치 → 집기
from od_msg.srv import SrvDepthPosition  # 약 이름 → 3D 좌표

# ──────────────────────────────
# 약국 전체 제어 노드
# ──────────────────────────────
class PharmacyManager(Node):
    """
    약국 자동화 시스템의 중앙 제어 노드
    - 음성으로부터 증상 또는 약 이름 수신
    - 증상이면 OpenAI로 약 추천
    - 약 이름이면 3D 위치 추정 → 로봇팔 제어
    """
    def __init__(self):
        super().__init__('pharmacy_manager')

        # 약 이름 리스트 (직접 언급 시 바로 집기 처리)
        self.available_drugs = [
            "모드콜", "콜대원", "하이펜", "타이레놀", "다제스",
            "락토프린", "포비돈", "미니온", "퓨어밴드", "Rohto C3 Cube"
        ]

        # 1. 증상/약 이름 텍스트 구독
        self.subscription = self.create_subscription(
            String,
            '/symptom_text',
            self.symptom_callback,
            10
        )

        # 2. 서비스 클라이언트 생성
        self.get_medicine_client = self.create_client(GetMedicineName, '/get_medicine_name')
        self.detect_position_client = self.create_client(SrvDepthPosition, '/get_3d_position')
        self.pickup_client = self.create_client(PickupMedicine, '/pickup_medicine')

        self.get_logger().info("Pharmacy Manager 실행됨. 사용자 입력을 기다리는 중...")

    def symptom_callback(self, msg: String):
        """ /symptom_text 토픽 수신 시 실행되는 콜백 함수 """
        symptom_text = msg.data.strip()
        self.get_logger().info(f"증상 또는 약 이름 수신: \"{symptom_text}\"")

        # case 1: 약 이름이면 → 위치 추정 → 집기
        if symptom_text in self.available_drugs:
            self.get_logger().info("직접 약 이름이 감지됨 → 바로 위치 추정 및 집기")
            position = self.call_detect_position(symptom_text)
            if position:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = position
                pose.orientation.w = 1.0
                success = self.call_pickup(pose)
                if success:
                    self.get_logger().info("약 집기 성공")
                else:
                    self.get_logger().error("약 집기 실패")
            else:
                self.get_logger().error("약 위치 감지 실패")
            return

        # case 2: 일반 증상이면 → 약 추천 → 위치 추정 → 집기
        medicine_name = self.call_get_medicine_name(symptom_text)
        if not medicine_name:
            self.get_logger().error("약 이름 추출 실패")
            return

        self.get_logger().info(f"추천된 약: {medicine_name}")
        position = self.call_detect_position(medicine_name)
        if not position:
            self.get_logger().error("약 위치 감지 실패")
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
        """ 증상 텍스트를 OpenAI 기반 서비스로 전송 → 약 이름 반환 """
        if not self.get_medicine_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/get_medicine_name 서비스 없음")
            return None

        request = GetMedicineName.Request()
        request.symptom = symptom
        future = self.get_medicine_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().medicine if future.result() else None

    def call_detect_position(self, medicine: str):
        """ YOLO 기반 위치 감지 서비스 호출 → 약 이름으로 3D 위치 추정 """
        if not self.detect_position_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/get_3d_position 서비스 없음")
            return None

        request = SrvDepthPosition.Request()
        request.target = medicine
        future = self.detect_position_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if not future.result():
            return None

        pos = future.result().depth_position
        if sum(pos) == 0.0:
            return None
        return pos

    def call_pickup(self, pose: Pose) -> bool:
        """ 로봇팔 제어 서비스 호출 → 약 집기 수행 """
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/pickup_medicine 서비스 없음")
            return False

        request = PickupMedicine.Request()
        request.pose = pose
        future = self.pickup_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success if future.result() else False

# ──────────────────────────────
# main 실행
# ──────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PharmacyManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
