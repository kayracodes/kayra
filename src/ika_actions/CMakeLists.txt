cmake_minimum_required(VERSION 3.8)
project(ika_actions)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ika_msgs REQUIRED)
# generate action interfaces
set(action_files
  "action/RampAlignment.action"
  "action/GoForward.action"
  "action/AlignWithPath.action"
  "action/LockTarget.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  DEPENDENCIES builtin_interfaces action_msgs
)

# Install Python executables
install(PROGRAMS
  ika_actions/align_ramp.py
  ika_actions/go_forward.py
  ika_actions/align_with_path.py
  ika_actions/lock_target.py
  ika_actions/lock_target_real.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
