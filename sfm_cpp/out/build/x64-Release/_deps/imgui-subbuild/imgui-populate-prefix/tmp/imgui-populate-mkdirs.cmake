# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-src")
  file(MAKE_DIRECTORY "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-src")
endif()
file(MAKE_DIRECTORY
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-build"
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix"
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/tmp"
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp"
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/src"
  "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/build/x64-Release/_deps/imgui-subbuild/imgui-populate-prefix/src/imgui-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
