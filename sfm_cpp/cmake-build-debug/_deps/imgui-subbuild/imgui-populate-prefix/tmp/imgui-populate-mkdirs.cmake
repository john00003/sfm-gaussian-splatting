# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-src"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-build"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/tmp"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/src"
  "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/github repos/sfm-gaussian-splatting/sfm_cpp/cmake-build-debug/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
