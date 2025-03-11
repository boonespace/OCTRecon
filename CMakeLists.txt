cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("OCTRecon")

include_directories("include" "third_party")

if(WIN32)
    # opencv
    set(OpenCV_DIR "C:/Develop/opencv/build/x64/vc16/lib")
    # armadillo
    set(ARMADILLO_INCLUDE_DIRS "C:/Develop/armadillo/armadillo/include")
    set(ARMADILLO_LIBS "C:/Develop/armadillo/armadillo/examples/lib_win64/libopenblas.lib")
    include_directories(${ARMADILLO_INCLUDE_DIRS})
    # fftw
    set(FFTW_LIB ${CMAKE_SOURCE_DIR}/third_party/fftw3/lib/libfftw3-3.lib)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES
    include/DataStorage.cpp
    include/FFT.cpp
    include/Encoding.cpp
)

if(WIN32)
    set(LIBRARIES ${OpenCV_LIBS} ${ARMADILLO_LIBS} ${FFTW_LIB})
else()
    set(LIBRARIES ${OpenCV_LIBS} armadillo fftw3)
endif()

add_executable(octrecon OCTRecon.cpp ${SOURCES})

target_link_libraries(octrecon ${LIBRARIES})

if(WIN32)
    set(DLL_FILES
        ${CMAKE_SOURCE_DIR}/third_party/fftw3/dll/libfftw3-3.dll
        C:/Develop/opencv/build/x64/vc16/bin/opencv_world4100.dll
        C:/Develop/armadillo/armadillo/examples/lib_win64/libopenblas.dll
    )
    foreach(DLL ${DLL_FILES})
        add_custom_command(
            TARGET octrecon POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DLL} $<TARGET_FILE_DIR:octrecon>
        )
    endforeach()
endif()
