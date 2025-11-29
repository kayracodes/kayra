cmake_minimum_required(VERSION 3.8)
project(rake_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

# generate messages
set(msg_files
  "msg/RoadSign.msg"
  "msg/RoadSignArray.msg"
  "msg/MotorFeedback.msg"
  "msg/DeviceState.msg"
  "msg/ConfigUpdated.msg"
  "msg/SystemState.msg"
  "msg/ShootCommand.msg"
  "msg/ResetCommand.msg"
)

#generic service files
set(srv_files
  "srv/UpdateDeviceState.srv"
  "srv/UpdateConfig.srv"
  "srv/UpdateSystemState.srv"
  "srv/WaitForEvent.srv"
  "srv/LoadConfig.srv"
  "srv/SaveConfig.srv"
  "srv/LoadMission.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces
)

ament_package()
