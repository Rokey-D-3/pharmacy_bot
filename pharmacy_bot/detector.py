# ROS2 및 센서 메시지, 이미지 변환 관련 모듈
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# YOLO
from ultralytics import YOLO

import numpy as np
import time
import json
import os

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition

# ──────────────────────── ImgNode (Realsense 입력 처리) ────────────────────────
class ImgNode(Node):
    """
    RealSense 카메라의 색상/깊이 이미지와 내부 파라미터(CameraInfo)를 수신하는 노드
    """
    def __init__(self):
        super().__init__('img_node')
        self.bridge = CvBridge()
        self.color_frame = None
        self.color_frame_stamp = None
        self.depth_frame = None
        self.intrinsics = None

        # ROS2 토픽 구독 설정
        self.color_subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)

    def camera_info_callback(self, msg):
        """CameraInfo 메시지에서 Intrinsic 파라미터 추출"""
        self.intrinsics = {
            "fx": msg.k[0],
            "fy": msg.k[4],
            "ppx": msg.k[2],
            "ppy": msg.k[5]
        }

    def color_callback(self, msg):
        """Color 이미지 수신 및 변환"""
        self.color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.color_frame_stamp = str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec)

    def depth_callback(self, msg):
        """Depth 이미지 수신 및 변환"""
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_color_frame(self):
        return self.color_frame

    def get_color_frame_stamp(self):
        return self.color_frame_stamp

    def get_depth_frame(self):
        return self.depth_frame

    def get_camera_intrinsic(self):
        return self.intrinsics

# ──────────────────────── DetectorNode (YOLO 추론 + 3D 위치 계산) ────────────────────────
class DetectorNode(Node):
    """
    감지 요청이 들어오면 YOLO로 객체를 감지하고,
    중심 좌표의 depth 정보를 이용해 3D 위치를 계산하여 응답하는 서비스 노드
    """
    def __init__(self):
        super().__init__('object_detector')

        # Realsense 이미지 처리 노드 포함
        self.img_node = ImgNode()

        # YOLO 모델 로드
        self.model = self.load_yolo_model()

        # 카메라 intrinsics 확보 (fx, fy, ppx, ppy)
        self.intrinsics = self._wait_for_valid_data(self.img_node.get_camera_intrinsic, "intrinsics")

        # 서비스 서버 생성 : 약 이름 → 3D 위치 반환
        self.create_service(
            SrvDepthPosition,
            '/get_3d_position',
            self.handle_get_depth
            )

        self.get_logger().info("DetectorNode with YOLO + Realsense initialized.")

    def load_yolo_model(self):
        """
        YOLO 모델과 클래스 ID ↔ 이름 매핑 정보를 로드
        """
        package_path = get_package_share_directory("pharmacy_bot")
        model_path = os.path.join(package_path, "resource", "best.pt")
        class_path = os.path.join(package_path, "resource", "class_name.json")

        model = YOLO(model_path)

        # 클래스 이름 ↔ ID 매핑
        with open(class_path, "r", encoding="utf-8") as f:
            class_dict = json.load(f)
            reversed_class_dict = {v: int(k) for k, v in class_dict.items()}
        model.reversed_class_dict = reversed_class_dict
        return model

    def handle_get_depth(self, request, response):
        """
        클라이언트 요청: 약 이름(target)을 받고 해당 약의 3D 위치 반환
        """
        target = request.target.lower()
        self.get_logger().info(f"감지 요청: {target}")
        coords = self._compute_position(target)
        response.depth_position = [float(x) for x in coords]
        return response

    def _compute_position(self, target):
        """
        YOLO 감지 + 중심점 depth → 카메라 기준 3D 좌표 계산
        """
        rclpy.spin_once(self.img_node)
        frame = self.img_node.get_color_frame()
        if frame is None:
            self.get_logger().warn("No image frame available.")
            return 0.0, 0.0, 0.0
        
        # YOLO 추론
        results = self.model(frame, verbose=False)
        boxes = self._aggregate_detections(results)
        
        label_id = self.model.reversed_class_dict.get(target, -1)
        matches = [d for d in boxes if d["label"] == label_id]

        if not matches:
            self.get_logger().warn("No matching object detected.")
            return 0.0, 0.0, 0.0
        
        # 가장 신뢰도 높은 박스 선택
        best = max(matches, key=lambda x: x["score"])
        cx, cy = map(int, [(best["box"][0] + best["box"][2]) / 2, (best["box"][1] + best["box"][3]) / 2])
        
        # 중심 위치의 depth 값 얻기
        cz = self._get_depth(cx, cy)
        if cz is None:
            self.get_logger().warn("Depth unavailable.")
            return 0.0, 0.0, 0.0

        # 픽셀 좌표 + depth → 카메라 3D 좌표로 변환
        return self._pixel_to_camera_coords(cx, cy, cz)

    def _aggregate_detections(self, results, conf_thresh=0.5, iou_thresh=0.5):
        """
        YOLO 결과에서 score가 일정 이상인 것만 추출
        """
        raw = []
        for res in results:
            for box, score, label in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                if score >= conf_thresh:
                    raw.append({"box": box, "score": score, "label": int(label)})

        return raw

    def _get_depth(self, x, y):
        """픽셀 좌표의 depth 값을 안전하게 읽어옵니다."""
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth")
        try:
            return frame[y, x]
        except IndexError:
            return None

    def _pixel_to_camera_coords(self, x, y, z):
        """픽셀 좌표 + depth → 카메라 좌표계 변환 (x, y, z in meters)"""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z
        )

    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도"""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Waiting for {description}...")
            data = getter()
        return data


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

