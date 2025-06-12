# find_package(rosidl_default_generators REQUIRED)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "srv/GetMedicineName.srv"
#   "srv/DetectMedicinePose.srv"
#   "srv/PickupMedicine.srv"
# )

# ament_export_dependencies(rosidl_default_runtime)

cmake_minimum_required(VERSION 3.5)
project(pharmacy_bot)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetMedicineName.srv"
  "srv/PickupMedicine.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# Python 파일 수동 설치
install(
  DIRECTORY pharmacy_bot/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# 리소스 (모델, 클래스 등)
install(
  DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

# launch 파일 설치
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 메타데이터
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

