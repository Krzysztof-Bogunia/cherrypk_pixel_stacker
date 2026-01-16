cmake_minimum_required(VERSION 3.14)

project(global LANGUAGES CXX)

#global options
set(BUILD_SHARED_LIBS OFF)
set(BUILD_OBJECT_LIBS OFF)

#external projects options
set( EXTERNAL_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/src )
set( EXTERNAL_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/bin )
set(ExternalFlags 
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_TESTING:BOOL="0"
    -DBUILD_TESTS:BOOL="0"
    -DBUILD_SHARED_LIBS:BOOL="0"
    -DBUILD_PERF_TESTS:BOOL="0"
    -DBUILD_EXAMPLES:BOOL="0"
    -DBUILD_DOCS:BOOL="0"
    -DBUILD_TOOLS:BOOL="0"
    -DBUILD_OBJECT_LIBS:BOOL="0"
)

#boost options
set(BoostFlags 
    -DBOOST_USE_STATIC_LIBS:BOOL="1" 
    -DBOOST_RUNTIME_LINK=static
    -DBoost_USE_STATIC_RUNTIME:BOOL="1"
    -DBoost_ENABLE_CMAKE:BOOL="1"
    -DBoost_DEBUG:BOOL="0" 
    -DBOOST_BUILD_TESTS:BOOL="0"
    -DBOOST_BUILD_EXAMPLES:BOOL="0"
    -DBOOST_BUILD_DOCS:BOOL="0"
    -DBoost_FIND_REQUIRED:BOOL="1" 
    -DBoost_FIND_QUIETLY:BOOL="1" 
    -DBOOST_SKIP_INSTALL_RULES:BOOL="1"
    -DBOOST_TEST_NO_LIB:BOOL="1"
    -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL="1"
    -DCMAKE_CXX_EXTENSIONS:BOOL="1"
) 
list (APPEND BoostFlags ${ExternalFlags})

#eigen options
set(EigenFlags 
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DEIGEN_BUILD_TESTING:BOOL="0"
    -DEIGEN_MPL2_ONLY:BOOL="1"
    -DEIGEN_BUILD_PKGCONFIG:BOOL="0"
    -DEIGEN_BUILD_DOC:BOOL="0"  
) 
list (APPEND EigenFlags ${ExternalFlags})

include(ExternalProject)
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_DIR})

message(STATUS "ExternalProject Boost...")
set( Boost_Bootstrap_Command )
if( UNIX )
  set( Boost_Bootstrap_Command ./bootstrap.sh )
  set( Boost_b2_Command ./b2 )
else()
  if( WIN32 )
    set( Boost_Bootstrap_Command ./bootstrap.bat )
    set( Boost_b2_Command b2.exe )
  endif()
endif()
ExternalProject_Add(
    Boost
    PREFIX ${EXTERNAL_INSTALL_DIR}/Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.84.0
    INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
    CMAKE_ARGS ${BoostFlags}
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND ${Boost_Bootstrap_Command}
    BUILD_COMMAND  ${Boost_b2_Command} install
      --without-python
      --without-mpi
      --disable-icu
      --prefix=${EXTERNAL_INSTALL_DIR}
      --threading=single,multi
      --link=static
      --variant=release
      -j12
    INSTALL_COMMAND ""
)
include_directories(${EXTERNAL_SOURCE_DIR}/Boost)
include_directories(${EXTERNAL_SOURCE_DIR}/Boost/boost)
include_directories(${EXTERNAL_SOURCE_DIR}/Boost/boost/algorithm/cxx11)
include_directories(${EXTERNAL_INSTALL_DIR})

message(STATUS "ExternalProject Eigen3...")
ExternalProject_Add(
    Eigen3
    PREFIX ${EXTERNAL_INSTALL_DIR}/Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}/Eigen3
    INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
    CMAKE_ARGS ${EigenFlags}
)
include_directories(${EXTERNAL_SOURCE_DIR}/Eigen3)

message(STATUS "ExternalProject libInterpolate...")
ExternalProject_Add(
    libInterpolate
    PREFIX ${EXTERNAL_INSTALL_DIR}/libInterpolate
    GIT_REPOSITORY https://github.com/CD3/libInterpolate.git
    GIT_TAG 2.7.2
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}/libInterpolate
    INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
    CMAKE_ARGS ${ExternalFlags}
    DEPENDS Boost Eigen3
)
include_directories(${EXTERNAL_SOURCE_DIR}/libInterpolate)

ExternalProject_Add(
    src
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
    CMAKE_ARGS 
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/bin
      -DEXTERNAL_INSTALL_DIR=${EXTERNAL_INSTALL_DIR}
      -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_DIR}
    DEPENDS libInterpolate
)

