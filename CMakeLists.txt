cmake_minimum_required(VERSION 3.5)
project(pharmacy_bot)

# 필수 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# ───── 서비스 인터페이스 생성 ─────
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetMedicineName.srv"
  "srv/PickupMedicine.srv"
  "srv/SrvDepthPosition.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# ───── 실행 가능한 Python 노드 설치 ─────
install(
  PROGRAMS
    pharmacy_bot/pharmacy_manager.py
    pharmacy_bot/voice_input.py
    pharmacy_bot/symptom_matcher.py
    pharmacy_bot/detector.py
    pharmacy_bot/robot_arm.py
    pharmacy_bot/symptom_matcher_test.py
    pharmacy_bot/symptom_subscriber_test.py
    pharmacy_bot/voice_and_match_test.py
    pharmacy_bot/pharmacy_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

# ───── robot_control 서브모듈 설치 ─────
install(
  DIRECTORY pharmacy_bot/robot_control
  DESTINATION lib/${PROJECT_NAME}/pharmacy_bot
)

# ───── 리소스 (YOLO 모델, 클래스 파일 등) 설치 ─────
install(
  DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

# ───── launch 파일 설치 ─────
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ───── 패키지 메타데이터 설치 ─────
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# ───── 패키지 마무리 ─────
ament_package()

