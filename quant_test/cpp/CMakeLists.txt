message(STATUS "MNN Depends:" ${MNN_DEPS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} /usr/src/linux-headers-5.4.0-48-generic/include/config/squashfs/file/)
add_executable(retinanet_quant.out ${CMAKE_CURRENT_LIST_DIR}/retinanet_quant.cpp)
target_link_libraries(retinanet_quant.out ${MNN_DEPS} ${OpenCV_LIBS})

add_executable(ssd_quant.out ${CMAKE_CURRENT_LIST_DIR}/ssd_quant.cpp)
target_link_libraries(ssd_quant.out ${MNN_DEPS} ${OpenCV_LIBS})