cmake_minimum_required(VERSION 3.10)
project(SfmCeresDemo)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DGL_SILENCE_DEPRECATION)

include_directories(/usr/local/opt/exiv2/include)
link_directories(/usr/local/opt/exiv2/lib)

set(CMAKE_FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/.cache)

include(FetchContent)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG docking
)
FetchContent_MakeAvailable(imgui)

add_library(ImGui
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(ImGui PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Exiv2 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${EXIV2_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

add_executable(sfm_ceres
  src/main.cpp
  src/viewer.cpp
  src/gui.cpp
  src/sfm_system.cpp
)

target_link_libraries(sfm_ceres
  ${CERES_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Pangolin_LIBRARIES}
  OpenGL::GL
  glfw
  GLEW::GLEW
  ImGui
  exiv2
)

if (APPLE)
  if (EXISTS "/opt/homebrew/include/GLFW")
    target_include_directories(ImGui PUBLIC /opt/homebrew/include)
  elseif (EXISTS "/usr/local/include/GLFW")
    target_include_directories(ImGui PUBLIC /usr/local/include)
  else()
    message(FATAL_ERROR "GLFW not found. Please install with: brew install glfw")
  endif()
endif()
