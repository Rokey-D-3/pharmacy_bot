import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pharmacy_bot.srv import GetMedicineName, DetectMedicinePose, PickupMedicine
from geometry_msgs.msg import Pose

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose

# 서비스 인터페이스 불러오기
from srv import GetMedicineName          # 증상 → 약 이름
from od_msg.srv import SrvDepthPosition  # 약 이름 → 3D 위치
from srv import PickupMedicine           # 위치 → 집기

class PharmacyManager(Node):
    """
    전체 약국 자동화 시스템의 중앙 제어 노드
    증상 텍스트를 받아 약을 추천하고, 약의 위치를 감지해 로봇팔이 약을 집도록 제어함
    """
    def __init__(self):
        super().__init__('pharmacy_manager')

        # 1. 음성 인식 결과 구독: 증상 텍스트가 이 토픽으로 퍼블리시됨
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

        self.get_logger().info("Pharmacy Manager 실행됨. 증상 입력을 기다리는 중...")

    def symptom_callback(self, msg: String):
        symptom_text = msg.data
        self.get_logger().info(f"증상 수신: \"{symptom_text}\"")

        # 1. 증상 → 약 추천
        medicine_name = self.call_get_medicine_name(symptom_text)
        if not medicine_name:
            self.get_logger().error("약 이름 추출 실패")
            return

        self.get_logger().info(f"추천 약: {medicine_name}")

        # 2. 약 이름 → 3D 위치
        position = self.call_detect_position(medicine_name)
        if not position:
            self.get_logger().error("약 위치 감지 실패")
            return

        x, y, z = position
        self.get_logger().info(f"약 위치: x={x:.2f}, y={y:.2f}, z={z:.2f}")

        # 3. 위치 → 로봇팔로 집기 명령
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0  # 기본 방향

        result = self.call_pickup(pose)
        if result:
            self.get_logger().info("약 집기 성공")
        else:
            self.get_logger().error("약 집기 실패")

    def call_get_medicine_name(self, symptom: str) -> str:
        """
        증상 텍스트를 OpenAI 기반 symptom_matcher 서비스로 보내
        약 이름을 추출하는 함수
        """
        if not self.get_medicine_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/get_medicine_name 서비스 없음")
            return None

        request = GetMedicineName.Request()
        request.symptom = symptom
        future = self.get_medicine_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().medicine if future.result() else None

    def call_detect_position(self, medicine: str):
        """
        감지된 약 이름을 YOLO 기반 object detection 서비스로 보내
        3D 위치를 얻는 함수
        """
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
        """
        Pose를 로봇팔 제어 서비스로 보내 약을 집도록 하는 함수
        """
        if not self.pickup_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/pickup_medicine 서비스 없음")
            return False

        request = PickupMedicine.Request()
        request.pose = pose
        future = self.pickup_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success if future.result() else False


def main(args=None):
    rclpy.init(args=args)
    node = PharmacyManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
