CMAKE_MINIMUM_REQUIRED (VERSION 3.18)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT("OCTRecon")

INCLUDE_DIRECTORIES("include" "third_party")

# opencv
if (WIN32)
    SET(OpenCV_DIR "C:/Develop/opencv/build/x64/vc16/lib")
endif()
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# armadillo
SET(ARMADILLO_INCLUDE_DIRS "C:/Develop/armadillo/armadillo/include")
SET(ARMADILLO_LIBS "C:/Develop/armadillo/armadillo/examples/lib_win64/libopenblas.lib")
INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})

# fftw
set(FFTW_LIB ${CMAKE_SOURCE_DIR}/third_party/fftw3/lib/libfftw3-3.lib)

# 定义包含的源文件（仅有 .cpp 需要添加，头文件不需要）
set(SOURCES
    include/DataStorage.cpp
    include/FFT.cpp
)

# 定义需要包含的 DLL 文件
set(DLL_FILES 
    ${CMAKE_SOURCE_DIR}/third_party/fftw3/dll/libfftw3-3.dll
)

add_executable(octrecon OCTRecon.cpp ${SOURCES})
target_link_libraries(octrecon ${OpenCV_LIBS} ${FFTW_LIB})
foreach(DLL ${DLL_FILES})
    add_custom_command(TARGET octrecon POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL} $<TARGET_FILE_DIR:octrecon>
    )
endforeach()