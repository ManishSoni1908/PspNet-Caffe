cmake_minimum_required(VERSION 2.8.3)

#set(TARGET_BINARY_NAME ${CURRENT_TARGET})

project(pspnet_test)
#message(STATUS ${CUDA_NVCC_FLAGS})
#nvcc flags
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)


#file(GLOB SOURCES "sources/*.cpp")

find_package(Caffe REQUIRED HINTS "/home/manish/tcs/arc_setup/libraries_install/caffe")
find_package(CUDA REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  ${Caffe_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  kernels
  utility
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

include_directories(${PCL_INCLUDE_DIRS})

find_package(Boost REQUIRED filesystem system)
include_directories(${Boost_INCLUDE_DIRS} )

add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
add_definitions(${Caffe_DEFINITIONS})
add_definitions(${Caffe_DEFINITIONS})
add_definitions(${CUDA_DEFINITIONS})

add_executable(test sources/test.cpp)
target_link_libraries(test ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pixel_value sources/pixel_value.cpp)
target_link_libraries(pixel_value ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(ssd_test sources/ssd_test.cpp)
target_link_libraries(ssd_test ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(mask_save sources/mask_save.cpp)
target_link_libraries(mask_save ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(manual_mask_making sources/manual_mask_making.cpp)
target_link_libraries(manual_mask_making ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(distinct_random sources/distinct_random.cpp)
target_link_libraries(distinct_random ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(test_crf sources/test_crf.cpp)
target_link_libraries(test_crf ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(write_list sources/write_list.cpp)
target_link_libraries(write_list ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(pcd_test sources/pcd_test.cpp)
target_link_libraries(pcd_test ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})


add_executable(crop sources/crop.cpp)
target_link_libraries(crop ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(data_pspnet_format sources/data_pspnet_format.cpp)
target_link_libraries(data_pspnet_format ${OpenCV_LIBRARIES})


add_executable(annotation sources/annotation.cpp)
target_link_libraries(annotation ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(annotation_crf sources/annotation_crf.cpp)
target_link_libraries(annotation_crf ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(vecadd sources/vecadd.cpp)
target_link_libraries(vecadd ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(clutter_generation sources/clutter_generation.cpp)
target_link_libraries(clutter_generation ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})



add_executable(gilete_test sources/gillete_test.cpp)
target_link_libraries(gilete_test ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})



add_executable(encon sources/encon.cpp)
target_link_libraries(encon ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})

#add_executable(image_conv sources/image_conv.cpp)
#target_link_libraries(image_conv ${Caffe_LIBRARIES} ${OpenCV_LIBRARIES})


#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
##LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
#set(CUDA_SEPARABLE_COMPILATION ON)

#set(CUDA_SOURCES sources/cuda_test.cu)
#CUDA_WRAP_SRCS(cuda_test_main OBJ CUDA_OBJECT_FILES ${CUDA_SOURCES} STATIC)
#CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(CUDA_OBJECT_FILE_NAMES cuda_test_main ${CUDA_NVCC_FLAGS} ${CUDA_OBJECT_FILES} )



#message("hi")
#message("hi")
#message("hi")
#message("hi")
#message(STATUS ${CUDA_SOURCES})
#message("hi")
#message(STATUS ${CUDA_OBJECT_FILES})
#message("hi")
#message(STATUS ${CUDA_OBJECT_FILE_NAMES})
#message("hi")
#message(STATUS ${CUDA_CUBLAS_LIBRARIES})
#message("hi")
#message("hi")



#add_executable(www  ${CUDA_OBJECT_FILES} ${CUDA_OBJECT_FILE_NAMES} ${CUDA_SOURCES})
#set_target_properties(www PROPERTIES LINKER_LANGUAGE CXX)


#set(CUDA_OPTIONS)
#CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS(${CUDA_OBJECT_FILE_NAMES} www "${CUDA_OPTIONS}" ${CUDA_OBJECT_FILES})


#set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})


#target_link_libraries(www ${CUDA_LIBS} ${OpenCV_LIBRARIES})


#file(GLOB  cu  *.cu)
#set(cuda_sources sources/cuda_test.cu)


#message("hi")
#messege("hi")
#messege(STATUS ${cu})
#messege("hi")
#messege("hi")



#cuda_add_executable(preProcess ${cu})
#target_link_libraries(preProcess ${OpenCV_LIBS})

#CUDA_ADD_EXECUTABLE(cuda_test_cmake sources/cuda_test.cu)
#CUDA_ADD_EXECUTABLE(image_conv1 sources/image_conv.cu)
#TARGET_LINK_LIBRARIES()
#cuda_add_executable(cuda_test_cmake2 sources/cuda_test2.cu)
#target_link_libraries(cuda_test_cmake2  ${CUDA_LIBRARIES})
