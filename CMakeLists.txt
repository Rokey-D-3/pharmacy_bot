find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetMedicineName.srv"
  "srv/DetectMedicinePose.srv"
  "srv/PickupMedicine.srv"
)

ament_export_dependencies(rosidl_default_runtime)