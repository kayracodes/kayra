cmake_minimum_required(VERSION 3.8)
project(rake_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rake_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    ${PROJECT_NAME}/constants.py
    ${PROJECT_NAME}/states.py
    ${PROJECT_NAME}/node.py
    ${PROJECT_NAME}/node1.py
    ${PROJECT_NAME}/node2.py
    ${PROJECT_NAME}/core.py
    DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
