message(STATUS "MNN Depends:" ${MNN_DEPS})
add_executable(pictureRecognition.out ${CMAKE_CURRENT_LIST_DIR}/pictureRecognition.cpp)
target_link_libraries(pictureRecognition.out ${MNN_DEPS})
add_executable(pictureRotate.out ${CMAKE_CURRENT_LIST_DIR}/pictureRotate.cpp)
target_link_libraries(pictureRotate.out ${MNN_DEPS})

add_executable(multiPose.out ${CMAKE_CURRENT_LIST_DIR}/multiPose.cpp)
target_link_libraries(multiPose.out ${MNN_DEPS})

add_executable(segment.out ${CMAKE_CURRENT_LIST_DIR}/segment.cpp)
target_link_libraries(segment.out ${MNN_DEPS})

add_executable(expressDemo.out ${CMAKE_CURRENT_LIST_DIR}/expressDemo.cpp)
target_link_libraries(expressDemo.out ${MNN_DEPS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(retinanet.out ${CMAKE_CURRENT_LIST_DIR}/retinanet.cpp)
target_link_libraries(retinanet.out ${MNN_DEPS} ${OpenCV_LIBS})

add_executable(ssd.out ${CMAKE_CURRENT_LIST_DIR}/ssd.cpp)
target_link_libraries(ssd.out ${MNN_DEPS} ${OpenCV_LIBS})

add_executable(classification.out ${CMAKE_CURRENT_LIST_DIR}/classification.cpp)
target_link_libraries(classification.out ${MNN_DEPS} ${OpenCV_LIBS})

add_executable(retinaface.out ${CMAKE_CURRENT_LIST_DIR}/retinaface.cpp)
target_link_libraries(retinaface.out ${MNN_DEPS} ${OpenCV_LIBS})

add_executable(face_recognition.out ${CMAKE_CURRENT_LIST_DIR}/face_recognition.cpp)
target_link_libraries(face_recognition.out ${MNN_DEPS} ${OpenCV_LIBS})