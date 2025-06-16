#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from pharmacy_bot.srv import PickupMedicine  # Point → bool 응답 서비스

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# OnRobot RG2 그리퍼 제어용 클래스
from robot_control.onrobot import RG

# DSR 로봇 API
import DR_init

import rclpy
from rclpy.node import Node



# ─────────────── DSR API 초기화 ───────────────

DR_init.__dsr__id = "dsr01"
DR_init.__dsr__model = "m0609"
rclpy.init()

DR_init.__dsr__node = rclpy.create_node("robot_arm_node", namespace=DR_init.__dsr__id)


from DSR_ROBOT2 import movej, movel, get_current_posx, mwait, set_tool, set_tcp, posx


# ─────────────── 로봇 및 그리퍼 설정 ───────────────

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY = 20     # 이동 속도
ACC = 20          # 가속도

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
        # self.pickup_callback()
    def init_robot(self):
        """
        로봇팔을 초기 관절 자세로 이동시키고 그리퍼를 열어서 준비시킴
        """
        set_tool("TW_brkt")       # 사전에 등록된 툴 이름
        set_tcp("RG2_brkt")       # 사전에 등록된 TCP 이름
        home_pose=[-140.48, -34.46, 146, 95.15, 49.70, -19.84]  # 홈 위치 조인트값
        JReady = [0, 0, 90, 0, 90, 0]  # 기본 posture (rad 아님)
        movej(home_pose, vel=VELOCITY, acc=ACC)
        gripper.open_gripper()
        mwait()


    def pickup_callback(self, request, response):
    # def pickup_callback(self):
        self.get_logger().info("서비스 요청 수신됨")
        """
        서비스 콜백:
        point 기반으로 좌표 변환 및 보정
        약 위치로 로봇팔 이동
        그리퍼로 약 집기 -> 성공 여부 반환
        약을 집은 후 다시 초기 위치로 복귀
        """


        point = request.point
        width = request.width
        print('point : ', point)
        print('width : ',width)
        # ───── 위치 보정 (ROS는 m, 로봇은 mm 단위) ─────
        x = point.position.x                    # x는 그대로
        y = point.position.y   # 벽면에 붙은 약 → y+에서 접근
        z = point.position.z  
        # x = point.position.x                    # x는 그대로
        # y = point.position.y + WALL_APPROACH_OFFSET  # 벽면에 붙은 약 → y+에서 접근
        # z = point.position.z  + DEPTH_OFFSET      # z축 살짝 내려줌
        # z = max(z, MIN_DEPTH )                   # 너무 낮아지는 것 방지

        # ───── Orientation 설정 (그리퍼가 -y 방향을 보게 설정) ─────
        # ZYZ 오일러 각도: 그리퍼가 정면(벽 쪽)을 향하게 설정
        # r = R.from_euler("ZYZ", [90, -90, -90], degrees=True)
        # rx, ry, rz = r.as_euler("ZYZ", degrees=True)

        # 최종 타겟 Point
        target_pos = [x, y, z, 90, -90, -90]
        # 실제로 약 박스 집기 위해서는 y -30 더 해야할듯
        self.get_logger().info(f"약 위치로 이동 중: {target_pos[:3]}")
        self.get_logger().info(f"약 폭(mm)에 맞춰 그리퍼 조절 중: {width:.1f} mm")


        #####
        # target_pos = [-90.5, -27.624, 678.0, 89.999, 90.0, 0.0]
        safe_pos = posx(300, 0, 300, 0, 180, 0)

        try:
            # ───── 이동 및 집기 동작 수행 ─────
            #gripper move 추가 필요
            print("이동 전 디버깅 target_pos : ",target_pos)
            
            # target_pos = posx(target_pos[0],target_pos[1],target_pos[2],target_pos[3],target_pos[4],target_pos[5])
            # movel(target_pos, vel=VELOCITY, acc=ACC)
            middle_point=[-78.95, 22.08, 107.49, 17.03, -40.87, 76.95]
            movej(middle_point,vel=VELOCITY,acc=ACC)
            mwait()
            ret = movel(target_pos, vel=VELOCITY, acc=ACC)
            print("movel 결과:", ret)
            # print('이동직전')
            mwait()
            # print('이동직후')

            gripper.close_gripper(100)  # 집기
            while gripper.get_status()[0]:  # busy 상태일 때 대기
                time.sleep(0.5)
            mwait()

            gripper.move_gripper(width_val=int(width))   # 놓기
            while gripper.get_status()[0]:
                time.sleep(0.5)
            mwait()

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

