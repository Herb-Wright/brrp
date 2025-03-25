cmake_minimum_required(VERSION 3.0.2)
project(custom_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  sensor_msgs
  shape_msgs
)

# Add the custom message files
add_message_files(
  FILES
  MeshList.msg
)

# Generate messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  shape_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs shape_msgs
)