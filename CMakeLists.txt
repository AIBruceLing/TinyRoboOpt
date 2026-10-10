cmake_minimum_required(VERSION 3.14)
project(TinyRoboOpt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 开启常用优化与警告
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 寻找系统中的 Eigen3 库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

message(STATUS ">>> Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")

# 注册第 1 阶段的目标可执行文件 (后续有新代码在此追加即可)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/01_line_search/backtracking_line_search_armijo.cpp")
    add_executable(backtracking_line_search_armijo 01_line_search/backtracking_line_search_armijo.cpp)
    target_link_libraries(backtracking_line_search_armijo PRIVATE Eigen3::Eigen)
endif()