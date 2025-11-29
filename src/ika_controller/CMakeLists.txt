cmake_minimum_required(VERSION 3.8)
project(ika_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rake_core REQUIRED)
find_package(rake_msgs REQUIRED)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config mission
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    ${PROJECT_NAME}/key_vel_controller.py
    ${PROJECT_NAME}/controller.py
    ${PROJECT_NAME}/joy_controller.py
    ${PROJECT_NAME}/joy_controller_state.py
    ${PROJECT_NAME}/event_detector.py
    ${PROJECT_NAME}/event_caller.py
    ${PROJECT_NAME}/serial_controller.py
    ${PROJECT_NAME}/serial_controller_real.py
    ${PROJECT_NAME}/camera_node.py
    ${PROJECT_NAME}/camera_node_real.py
    ${PROJECT_NAME}/mission_planner.py
    DESTINATION lib/${PROJECT_NAME}/
)
ament_package()
