cmake_minimum_required(VERSION 3.8)
project(ika_msgs)

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
  "msg/ColorFilter.msg"
  "msg/CameraShooterDebug.msg"
  "msg/MotorFeedback.msg"
  "msg/RampFeedback.msg"
  "msg/RampDistance.msg"
  "msg/RampInfo.msg"
  "msg/IsInWater.msg"
)

#generic service files
# set(srv_files
#   "srv/UpdateConfig.srv"
#   "srv/UpdateSystemState.srv"
# )

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  # ${srv_files}
  DEPENDENCIES builtin_interfaces
)

ament_package()
