cmake_minimum_required(VERSION 2.8.3)

#set(TARGET_BINARY_NAME ${CURRENT_TARGET})

project(pspnet_test)


find_package(Caffe REQUIRED HINTS "/home/manish/tcs/arc_setup/libraries_install/caffe")
find_package(CUDA REQUIRED)

include_directories(
  ${Caffe_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}

)
message("hi")
message("hi")
message("hi")
message("hi")
message(STATUS ${CUDA_LIBRARIES})
message(STATUS ${Caffe_LIBRARIES})
message(STATUS ${CUDA_INCLUDE_DIRS})
message(STATUS ${CUDA_NVCC_FLAGS})
message("hi")
message("hi")
message("hi")



find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(Boost REQUIRED filesystem system)
include_directories(${Boost_INCLUDE_DIRS} )

add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
add_definitions(${Caffe_DEFINITIONS})
add_definitions(${Caffe_DEFINITIONS})
add_definitions(${CUDA_DEFINITIONS})

add_executable(test sources/test.cpp)
target_link_libraries(test ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(test_crf sources/test_crf.cpp)
target_link_libraries(test_crf ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})




