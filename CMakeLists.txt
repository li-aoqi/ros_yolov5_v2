cmake_minimum_required(VERSION 3.5)
project(ros_yolov5_v2)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  geometry_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS src/laq_debug_cpu.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
