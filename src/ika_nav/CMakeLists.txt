cmake_minimum_required(VERSION 3.8)
project(ika_nav)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ika_utils REQUIRED)
find_package(rake_msgs REQUIRED)
find_package(rake_core REQUIRED)
ament_python_install_package(${PROJECT_NAME})

# install(
#   DIRECTORY config
#   DESTINATION share/${PROJECT_NAME}
# )

install(PROGRAMS
    ${PROJECT_NAME}/path_planner.py
    ${PROJECT_NAME}/path_planner_real.py
    ${PROJECT_NAME}/path_tracker.py
    ${PROJECT_NAME}/path_resolver.py
    DESTINATION lib/${PROJECT_NAME}/
)
ament_package()
