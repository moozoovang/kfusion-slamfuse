cmake_minimum_required(VERSION 2.8)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3                  -Wall -Wno-unknown-pragmas -Wextra -Werror -Wno-error=deprecated-declarations -Wno-error=array-bounds")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3  --std=c++11 -Wall -Wno-unknown-pragmas -Wextra -Werror -Wno-error=deprecated-declarations -Wno-error=array-bounds ")

IF(ANDROID)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   -Wno-error ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error  ")
ENDIF()


find_package(TooN   REQUIRED)
find_package(Eigen3   REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${TOON_INCLUDE_PATHS})


if(APPLE)
  set(common_libraries     stdc++ pthread)
else(APPLE)
  if(ANDROID)
    set(common_libraries      stdc++  android log)
  else (ANDROID)
    set(common_libraries     rt stdc++ pthread)
  endif(ANDROID)
endif(APPLE)




include_directories(${SLAMBENCH_INCLUDE_DIR})
include_directories(./)
include_directories(./include)
include_directories(./thirdparty)
   
 # ----------------- CPP VERSION ----------------- 

add_library(${appname}-cpp kernels.cpp kfusion_library.cpp)
target_link_libraries(${appname}-cpp   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-cpp PROPERTIES COMPILE_FLAGS "-fPIC")
install(TARGETS ${appname}-cpp  DESTINATION  lib/)   

 # ----------------- OPENMP VERSION ----------------- 

add_library(${appname}-openmp  kernels.cpp kfusion_library.cpp)
target_link_libraries(${appname}-openmp   ${common_libraries})	
SET_TARGET_PROPERTIES(${appname}-openmp PROPERTIES COMPILE_FLAGS "-fPIC -fopenmp")
install(TARGETS ${appname}-openmp  DESTINATION  lib/)


 # ----------------- DYNAMIC  ----------------- 
 
generate_slam_library(${appname}-cpp "")
generate_slam_library(${appname}-openmp "-lgomp -fopenmp")


 #  ----------------- OCL VERSION ----------------- 
 
find_package(OpenCL)
   
if (OPENCL_FOUND) 
   include_directories(${OPENCL_INCLUDE_DIRS})
    add_library(${appname}-opencl  kernels_ocl.cpp common_opencl.cpp  kfusion_library.cpp)
    target_link_libraries(${appname}-opencl   ${common_libraries} ${OPENCL_LIBRARIES})	
    SET_TARGET_PROPERTIES(${appname}-opencl PROPERTIES COMPILE_FLAGS "-fPIC")
    
    install(TARGETS ${appname}-opencl  DESTINATION  lib/)
 

    generate_slam_library(${appname}-opencl "")

ENDIF()


 #  ----------------- CUDA VERSION ----------------- 

find_package(CUDA)
STRING(REPLACE "_static.a" ".so" CUDA_LIBRARIES "${CUDA_LIBRARIES}")


if (CUDA_FOUND) 

    SET( CUDA_PROPAGATE_HOST_FLAGS OFF ) 

    set(kfusion_cuda_srcs kernels.cu)
	    	
	cuda_add_library(${appname}-cuda    SHARED  ${kfusion_cuda_srcs}  kfusion_library.cpp OPTIONS -O3 -std=c++11 -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES)
	target_link_libraries(${appname}-cuda   ${common_libraries})	
    install(TARGETS ${appname}-cuda  DESTINATION  lib/)

	generate_slam_library(${appname}-cuda "")
	
endif(CUDA_FOUND)




