cmake_minimum_required(VERSION 2.8)
add_executable(rs-colorize rs-colorize/rs-colorize.cpp)

# Project name set
project( rs-colorize )

# Prohibit to generate MinSizeRel and RelWithDebInfo
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

#Add compiler options here
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# find OpenCV package and set paths
find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rs-colorize ${OpenCV_LIBS})

# for windows, default RealSense SDK install folder is set
# please chenge this location if it is installed in other folder.
if (WIN32)

include_directories( include C:/Github/forked/librealsense/include/ )
target_include_directories( rs-colorize PRIVATE C:/Github/forked/librealsense/include/librealsense2 )
target_link_libraries( rs-colorize C:/Github/forked/librealsense/build/Release/realsense2.lib )

# include_directories( include C:/Program\ Files\ \(x86\)/Intel\ RealSense\ SDK\ 2.0/include )
# target_include_directories( rs-colorize PRIVATE C:/Program\ Files\ \(x86\)/Intel\ RealSense\ SDK\ 2.0/include )
# target_link_libraries( rs-colorize C:/Program\ Files\ \(x86\)/Intel\ RealSense\ SDK\ 2.0/lib/x64/realsense2.lib )
endif (WIN32)

if (UNIX)
# find librealsense package and set paths
find_package(realsense2 REQUIRED )
include_directories(include ${realsense_INCLUDE_DIR})
target_include_directories(rs-colorize PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(rs-colorize ${realsense2_LIBRARY})
endif (UNIX)