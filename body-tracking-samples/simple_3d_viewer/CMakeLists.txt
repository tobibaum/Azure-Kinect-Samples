# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(k4arecord REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(simple_3d_viewer main.cpp skel_cap.cpp)

target_include_directories(simple_3d_viewer PRIVATE ../sample_helper_includes)
target_include_directories(simple_3d_viewer PRIVATE /usr/include/eigen3)

# Dependencies of this library
target_link_libraries(simple_3d_viewer PRIVATE 
    k4a
    k4abt
    k4arecord
    window_controller_3d::window_controller_3d
    ${OpenCV_LIBS}
    )
