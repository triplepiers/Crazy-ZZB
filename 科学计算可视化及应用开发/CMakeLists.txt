cmake_minimum_required(VERSION 3.30)

project(sample)

find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})

# IF NOT FOUND VTK, SKIP
if (NOT VTK_FOUND)
    message("${VTK_NOT_FOUND_MESSAGE}")
    return()
endif()

# IF FOUND, PRINT VTK VERSION
message (STATUS "VTK_VERSION: ${VTK_VERSION}")

# OLD VERSION
if (VTK_VERSION VERSION_LESS "3.10")
    include(${VTK_USE_FILE})
endif()

add_executable(${PROJECT_NAME} sample.cpp)
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
