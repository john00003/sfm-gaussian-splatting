cmake_minimum_required(VERSION 3.10)
project(PangolinTest)

set(CMAKE_CXX_STANDARD 14)

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${Pangolin_LIBRARIES})
