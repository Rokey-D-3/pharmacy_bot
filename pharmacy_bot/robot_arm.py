import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from pharmacy_bot.srv import PickupMedicine  # pose → bool 응답 서비스

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# OnRobot RG2 그리퍼 제어용 클래스
from robot_control.onrobot import RG

# DSR 로봇 API
import DR_init
from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, set_tool, set_tcp


# ─────────────── 로봇 및 그리퍼 설정 ───────────────

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY = 60     # 이동 속도
ACC = 60          # 가속도

# 약간의 z축 보정 및 안전 높이 제한
DEPTH_OFFSET = -5.0      # 약을 집기 위해 z축을 조금 낮춤 (mm)
MIN_DEPTH = 2.0          # 최소 안전 거리 (cm)

# 벽면 진열된 약에 접근하기 위한 y축 방향 보정
WALL_APPROACH_OFFSET = 100  # mm 만큼 y+에서 y- 방향으로 접근

# OnRobot RG2 그리퍼 설정
GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"
gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ─────────────── DSR API 초기화 ───────────────
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
rclpy.init()  # ROS 2 초기화
dsr_node = rclpy.create_node("robot_arm_node", namespace=ROBOT_ID)
DR_init.__dsr__node = dsr_node


# ─────────────── ROS 2 노드 정의 ───────────────

class RobotArm(Node):
    """
    /pickup_medicine 서비스를 통해 약 위치(Pose)를 받아
    로봇팔이 이동 및 집기를 수행한 후 응답을 반환하는 ROS 2 노드
    """
    def __init__(self):
        super().__init__('robot_arm')

        # pickup_medicine 서비스 서버 생성
        self.srv = self.create_service(
            PickupMedicine,
            '/pickup_medicine',
            self.pickup_callback
        )

        self.get_logger().info('Robot Arm 서비스 시작됨 (/pickup_medicine)')

        # 로봇 초기 자세로 이동
        self.init_robot()

    def init_robot(self):
        """
        로봇팔을 초기 관절 자세로 이동시키고 그리퍼를 열어서 준비시킴
        """
        set_tool("TW_brkt")       # 사전에 등록된 툴 이름
        set_tcp("RG2_brkt")       # 사전에 등록된 TCP 이름

        JReady = [0, 0, 90, 0, 90, 0]  # 기본 posture (rad 아님)
        movej(JReady, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()

    def pickup_callback(self, request, response):
        """
        서비스 콜백:
        pose.position을 기반으로 좌표 변환 및 보정
        약 위치로 로봇팔 이동
        그리퍼로 약 집기 -> 성공 여부 반환
        약을 집은 후 다시 초기 위치로 복귀
        """
        pose = request.pose

        # ───── 위치 보정 (ROS는 m, 로봇은 mm 단위) ─────
        x = pose.position.x * 1000                     # x는 그대로
        y = pose.position.y * 1000 + WALL_APPROACH_OFFSET  # 벽면에 붙은 약 → y+에서 접근
        z = pose.position.z * 1000 + DEPTH_OFFSET      # z축 살짝 내려줌
        z = max(z, MIN_DEPTH * 1000)                   # 너무 낮아지는 것 방지

        # ───── Orientation 설정 (그리퍼가 -y 방향을 보게 설정) ─────
        # ZYZ 오일러 각도: 그리퍼가 정면(벽 쪽)을 향하게 설정
        r = R.from_euler("ZYZ", [90, 90, 0], degrees=True)
        rx, ry, rz = r.as_euler("ZYZ", degrees=True)

        # 최종 타겟 pose
        target_pos = [x, y, z, rx, ry, rz]

        self.get_logger().info(f"약 위치로 이동 중: {target_pos[:3]}")

        try:
            # ───── 이동 및 집기 동작 수행 ─────
            movel(target_pos, vel=VELOCITY, acc=ACC)
            mwait()

            gripper.close_gripper()  # 집기
            while gripper.get_status()[0]:  # busy 상태일 때 대기
                time.sleep(0.5)
            mwait()

            gripper.open_gripper()   # 놓기
            while gripper.get_status()[0]:
                time.sleep(0.5)

            response.success = True
            self.get_logger().info("약 집기 성공")
        except Exception as e:
            self.get_logger().error(f"약 집기 실패: {str(e)}")
            response.success = False

        # 초기 자세로 복귀
        self.init_robot()
        return response


# ─────────────── main ───────────────

def main(args=None):
    node = RobotArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("로봇 제어 노드 종료됨")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

