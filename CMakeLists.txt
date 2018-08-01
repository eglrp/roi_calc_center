cmake_minimum_required(VERSION 2.8)
project(gesture_adjustment)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(calc_label_center src/calc_label_center.cpp)
target_link_libraries(calc_label_center ${OpenCV_LIBS})
