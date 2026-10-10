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

# 注册第目标可执行文件 
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/01_line_search/backtracking_line_search_armijo.cpp")
    add_executable(backtracking_line_search_armijo 01_line_search/backtracking_line_search_armijo.cpp)
    target_link_libraries(backtracking_line_search_armijo PRIVATE Eigen3::Eigen)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/01_line_search/strong_wolfe.cpp")
    add_executable(strong_wolfe 01_line_search/strong_wolfe.cpp)
    target_link_libraries(strong_wolfe PRIVATE Eigen3::Eigen)
endif()



if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/02_unconstrained/newton_lm_armijo.cpp")
    add_executable(newton_lm_armijo 02_unconstrained/newton_lm_armijo.cpp)
    target_link_libraries(newton_lm_armijo PRIVATE Eigen3::Eigen)
endif()


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/02_unconstrained/gauss_newton_armijo.cpp")
    add_executable(gauss_newton_armijo 02_unconstrained/gauss_newton_armijo.cpp)
    target_link_libraries(gauss_newton_armijo PRIVATE Eigen3::Eigen)
endif()