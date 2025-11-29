cmake_minimum_required(VERSION 3.8)
project(ika_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ika_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rake_msgs REQUIRED)
find_package(rake_core REQUIRED)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch ai_models video
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  ${PROJECT_NAME}/video_recorder.py
  ${PROJECT_NAME}/video_publisher.py
  ${PROJECT_NAME}/shoot_controller.py
  ${PROJECT_NAME}/shoot_controller.py
  ${PROJECT_NAME}/shoot_test_real.py
  ${PROJECT_NAME}/vision_transformer.py
  ${PROJECT_NAME}/vision_transformer_real.py
  # ${PROJECT_NAME}/process_video.py
  ${PROJECT_NAME}/sign_detector.py
  ${PROJECT_NAME}/hsv_filter.py
  ${PROJECT_NAME}/hsv_filters.py
  ${PROJECT_NAME}/hsv_map.py
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
