cmake_minimum_required(VERSION 3.8)
project(ika_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch worlds models urdf rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
    ${PROJECT_NAME}/spawn_entity_qos.py
    DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
