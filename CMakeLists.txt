cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)

### llama.cpp version
set(LLAMA_VERSION_MAJOR 1)
set(LLAMA_VERSION_MINOR 0)
set(LLAMA_VERSION_PATCH 0)
set(LLAMA_VERSION_BASE "${LLAMA_VERSION_MAJOR}.${LLAMA_VERSION_MINOR}.${LLAMA_VERSION_PATCH}")

# whether this is a development/nightly build
# set this to OFF when making a release from a release tag (vX.Y.Z)
# ref: https://github.com/ggml-org/ggml/discussions/1579
option(LLAMA_BUILD_IS_DEV "llama: dev build" ON)

if (LLAMA_BUILD_IS_DEV)
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}-dev")
else()
    # TODO: check that the current commit is tagged correctly according to the version specified above
    set(LLAMA_VERSION "${LLAMA_VERSION_BASE}")
endif()

message(STATUS "llama.cpp version: ${LLAMA_VERSION}")

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)
else()
    set(LLAMA_STANDALONE OFF)
endif()


option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (LLAMA_STANDALONE)
    # enable parallel builds for msbuild
    list(APPEND CMAKE_VS_GLOBALS UseMultiToolTask=true)
    list(APPEND CMAKE_VS_GLOBALS EnforceProcessCountAcrossBuilds=true)
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

option(LLAMA_BUILD_TESTS "Build CLI inference tests" OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/win-app.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD)
    if (${OLD})
        set(NEW "${ARGV2}")
        if(NEW)
            message(${TYPE} "${OLD} is deprecated, use ${NEW} instead")
            set(${NEW} ON PARENT_SCOPE)
        else()
            message(${TYPE} "${OLD} is deprecated and will be ignored")
        endif()
    endif()
endfunction()

llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_CURL)

#
# compile options
#

# clang stores the modification time of the precompiled header sources inside the
# header and rejects it when they differ, so the timestamp is left out of it
add_compile_options(
    "$<$<COMPILE_LANG_AND_ID:C,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang,IntelLLVM>:SHELL:-Xclang -fno-pch-timestamp>")

#
# 3rd-party
#

set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
add_subdirectory(ggml)

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

add_subdirectory(vendor)

add_subdirectory(common)
add_subdirectory(tools/cli)

if (LLAMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

llama_apply_app_manifests()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(TARGETS llama LIBRARY PUBLIC_HEADER)

install(TARGETS llama-common LIBRARY)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        VERSION ${LLAMA_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama)

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
