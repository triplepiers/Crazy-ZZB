cmake_minimum_required(VERSION 3.10)

project(sample)

find_package(VTK REQUIRED)

# IF NOT FOUND VTK, SKIP
if (NOT VTK_FOUND)
    message("${VTK_NOT_FOUND_MESSAGE}")
    return()
endif()

# IF FOUND, PRINT VTK VERSION
message (STATUS "VTK_VERSION: ${VTK_VERSION}")

# OLD VERSION
if (VTK_VERSION VERSION_LESS "8.90.0")
    include(${VTK_USE_FILE})
endif()

# manual
#add_executable(${PROJECT_NAME} ./src/dcmReader.cpp)

# auto
include_directories(./src)
aux_source_directory(./src SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
#add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${SRC_LIST})  # for MaxOS

target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
