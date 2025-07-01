cmake_minimum_required(VERSION 3.13) #3.13

set(CPU_DISPATCH AVX,AVX2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
set(BUILD_TESTING Off)
set(BUILD_TESTS OFF CACHE INTERNAL "")
set(BUILD_SHARED_LIBS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOCS OFF)
set(BUILD_TOOLS OFF)
set(BUILD_OBJECT_LIBS OFF)

#opencv options
set(OPENCV_EXTRA_MODULES_PATH "")
set(OPENCV_GENERATE_PKGCONFIG OFF) #ON
set(OpenCV_STATIC ON)
set(ENABLE_PRECOMPILED_HEADERS OFF) #OFF
set(BUILD_JAVA OFF)
set(BUILD_JPEG ON) #ON
set(BUILD_OPENJPEG ON)
set(BUILD_PNG ON)
set(BUILD_JPEG_TURBO_DISABLE ON)
set(WITH_OPENMP OFF) #ON
set(WITH_OPENCL OFF)
set(OPENCV_DNN_OPENCL OFF)
set(WITH_JPEG ON) #ON
set(WITH_JASPER OFF)
set(WITH_TIFF OFF)
set(WITH_WEBP OFF)
set(WITH_OPENEXR OFF)
set(WITH_IPP OFF)
set(WITH_TBB OFF)
set(WITH_EIGEN OFF)
set(WITH_OPENEXR OFF)
set(WITH_FFMPEG OFF)
set(WITH_CUDA OFF)
set(WITH_GSTREAMER OFF)
set(WITH_GTK OFF)
set(WITH_VTK OFF)
set(WITH_QT OFF)
set(WITH_OPENGL OFF)
set(WITH_SDL OFF)
set(WITH_ITT OFF)
set(BUILD_WITH_ITT OFF)
set(BUILD_ITT OFF)
set(WITH_CAROTENE OFF)
set(WITH_LAPACK OFF)
set(WITH_OPENJPEG OFF) #OFF
set(WITH_V4L OFF)
set(WITH_FFMPEG OFF)
set(WITH_GSTREAMER OFF)
set(WITH_MSMF OFF)
set(WITH_DSHOW OFF)
set(WITH_AVFOUNDATION OFF)
set(WITH_VA OFF)
set(WITH_VA_INTEL OFF)
set(WITH_AVIF OFF)
set(VIDEOIO_ENABLE_PLUGINS OFF)
set(BUILD_opencv_python2 OFF)
set(BUILD_opencv_python3 OFF)
set(OPENCV_ENABLE_NONFREE OFF)
set(OPENCV_FORCE_3RDPARTY_BUILD OFF)
set(OPENCV_IPP_ENABLE_ALL OFF)
set(BUILD_opencv_apps OFF)
# set(BUILD_opencv_imgcodecs ON) #ON
set(BUILD_opencv_videoio OFF)
# set(BUILD_opencv_video ON)
set(BUILD_opencv_stitching OFF) #ON
set(BUILD_opencv_objdetect OFF)
set(BUILD_opencv_dnn ON) #OFF
set(BUILD_opencv_ml OFF)
set(BUILD_opencv_gapi OFF) #OFF
set(BUILD_opencv_alphamat OFF)
set(BUILD_opencv_aruco OFF)
set(BUILD_opencv_bgsegm OFF)
set(BUILD_opencv_bioinspired OFF)
set(BUILD_opencv_cannops OFF)
set(BUILD_opencv_ccalib OFF)
set(BUILD_opencv_cudaarithm OFF)
set(BUILD_opencv_cudabgsegm OFF)
set(BUILD_opencv_cudacodec OFF)
set(BUILD_opencv_cudafeatures2d OFF)
set(BUILD_opencv_cudafilters OFF)
set(BUILD_opencv_cudaimgproc OFF)
set(BUILD_opencv_cudalegacy OFF)
set(BUILD_opencv_cudaobjdetect OFF)
set(BUILD_opencv_cudaoptflow OFF)
set(BUILD_opencv_cudastereo OFF)
set(BUILD_opencv_cudawarping OFF)
set(BUILD_opencv_cudev OFF)
set(BUILD_opencv_cvv OFF)
set(BUILD_opencv_datasets OFF)
set(BUILD_opencv_dnn_objdetect OFF)
set(BUILD_opencv_dnn_superres OFF)
set(BUILD_opencv_dpm OFF)
set(BUILD_opencv_face OFF)
set(BUILD_opencv_fastcv OFF)
set(BUILD_opencv_freetype OFF)
set(BUILD_opencv_fuzzy OFF)
set(BUILD_opencv_hdf OFF)
set(BUILD_opencv_hfs OFF)
set(BUILD_opencv_img_hash OFF)
set(BUILD_opencv_intensity_transform OFF)
set(BUILD_opencv_julia OFF)
set(BUILD_opencv_line_descriptor OFF)
set(BUILD_opencv_mcc OFF)
set(BUILD_opencv_optflow OFF)
set(BUILD_opencv_ovis OFF)
set(BUILD_opencv_phase_unwrapping OFF)
set(BUILD_opencv_plot OFF)
set(BUILD_opencv_quality OFF)
set(BUILD_opencv_rapid OFF)
set(BUILD_opencv_reg OFF)
set(BUILD_opencv_rgbd OFF)
set(BUILD_opencv_saliency OFF)
set(BUILD_opencv_sfm OFF)
set(BUILD_opencv_shape OFF)
set(BUILD_opencv_signal OFF)
set(BUILD_opencv_stereo OFF)
set(BUILD_opencv_structured_light OFF)
set(BUILD_opencv_superres OFF)
set(BUILD_opencv_surface_matching OFF)
set(BUILD_opencv_text OFF)
set(BUILD_opencv_tracking OFF)
set(BUILD_opencv_videostab OFF)
set(BUILD_opencv_viz OFF)
set(BUILD_opencv_wechat_qrcode OFF)
set(BUILD_opencv_xfeatures2d OFF)
set(BUILD_opencv_ximgproc OFF)
set(BUILD_opencv_xobjdetect OFF)
set(BUILD_opencv_xphoto OFF)
set(BUILD_opencv_java_bindings_generator OFF)
set(BUILD_opencv_js_bindings_generator OFF)
set(BUILD_opencv_python_bindings_generator OFF)
set(BUILD_opencv_python_tests OFF)
set(DNN_ENABLE_PLUGINS OFF) #ON
set(OPENCV_DNN_TFLITE OFF)
 #opencv experimental options
set(BUILD_ZLIB ON)

project(cherrypk_pixel_stacker)


include(FetchContent)

FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.11.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(opencv)
include_directories(${OpenCV_SOURCE_DIR})

set(DJSONCPP_WITH_POST_BUILD_UNITTEST OFF)
set(DJSONCPP_WITH_PKGCONFIG_SUPPORT OFF)
set(DJSONCPP_WITH_TESTS OFF)
set(DJSONCPP_WITH_CMAKE_PACKAGE OFF)
FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.6
)
FetchContent_GetProperties(jsoncpp)
if(NOT jsoncpp_POPULATED)
  message(STATUS "Fetching jsoncpp...")
  FetchContent_MakeAvailable(jsoncpp)
  # add_subdirectory(${jsoncpp_SOURCE_DIR}/)
endif()

FetchContent_Declare(
  libInterpolate
  GIT_REPOSITORY https://github.com/CD3/libInterpolate.git
  GIT_TAG master
)
set(BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_GetProperties(libInterpolate)
if(NOT libInterpolate_POPULATED)
  message(STATUS "Fetching libInterpolate...")
  FetchContent_MakeAvailable(libInterpolate)
endif()


set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(.)

set (CMAKE_EXE_LINKER_FLAGS "--static")

add_executable(cherrypk_pixel_stacker main.cpp)

target_include_directories(cherrypk_pixel_stacker PRIVATE
        ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
        ${OPENCV_MODULE_opencv_core_LOCATION}/include
        ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include
        ${OPENCV_MODULE_opencv_photo_LOCATION}/include        
        ${OPENCV_MODULE_opencv_dnn_LOCATION}/include
        ${OPENCV_MODULE_opencv_imgcodecs_LOCATION}/include
        ${OPENCV_MODULE_opencv_highgui_LOCATION}/include        
        ${OPENCV_MODULE_opencv_flann_LOCATION}/include        
        ${OPENCV_MODULE_opencv_features2d_LOCATION}/include
        ${OPENCV_MODULE_opencv_video_LOCATION}/include        
        ${OPENCV_MODULE_opencv_calib3d_LOCATION}/include
        )

add_compile_options(
    "$<$<CONFIG:RELEASE>:-O3 -flto>"
    "$<$<CONFIG:DEBUG>:-Wall -Wextra>"
)

target_link_libraries(cherrypk_pixel_stacker 
  opencv_core
  opencv_imgproc
  opencv_photo 
  opencv_dnn 
  opencv_imgcodecs
  opencv_highgui 
  opencv_video 
  opencv_calib3d 
  opencv_flann 
  opencv_features2d 
  jsoncpp_static
  libInterpolate::Interpolate
  )

# #print all variables
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
