cmake_minimum_required(VERSION 3.18)

project(asvo_cuda LANGUAGES CXX CUDA)

file(GLOB headers "*.h")
file(GLOB sources "*.cpp" "*.cu")

if(MSVC)
	source_group("Header Files" FILES ${headers})
	source_group("Source Files" FILES ${sources})
endif()

add_executable(demo ${headers} ${sources})
set_property(TARGET demo PROPERTY CXX_STANDARD 14)
set_property(TARGET demo PROPERTY CUDA_ARCHITECTURES 50)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
target_include_directories(demo PRIVATE ${GLEW_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(demo PRIVATE OpenGL::GL GLUT::GLUT ${GLEW_LIBRARIES})