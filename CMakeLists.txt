# CMake configuration

cmake_minimum_required(VERSION 3.0.0)

# Project configuration

project(asvo_cuda)

file(GLOB headers "*.h")
file(GLOB sources "*.cpp" "*.cu")

if(MSVC)
	source_group("Header Files" FILES ${headers})
	source_group("Source Files" FILES ${sources})
endif()

find_package(CUDA REQUIRED)

# 3rd party dependencies

set(freeglut_inc_dir "" CACHE PATH
"Path to include directory of freeglut 3.0.0 or later. \
The include directory is the one containing the directory 'GL' (so you can write '#include <GL/...>').")
set(freeglut_lib_debug "" CACHE FILEPATH
"Path to freeglut.lib version 3.0.0 or later.")
set(freeglut_lib_release "" CACHE FILEPATH
"Path to freeglutd.lib version 3.0.0 or later.")

set(glew_inc_dir "" CACHE PATH
"Path to include directory of Glew 1.12.0 or later. \
The include directory is the one containing the directory 'GL' (so you can write '#include <GL/...>').")
set(glew_lib "" CACHE FILEPATH
"Path to glew32.lib release, single-context, dynamic, version 1.12.0 or later. \
Either 32bit or 64bit depending on how you intend to build this project.")

# Include paths

include_directories(
	${freeglut_inc_dir}
	${glew_inc_dir}
)

cuda_add_executable(demo ${headers} ${sources})

# Libraries

target_link_libraries(demo
	${glew_lib}
	debug ${freeglut_lib_debug}
	optimized ${freeglut_lib_release}
)