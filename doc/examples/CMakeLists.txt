add_executable(nvvl_loader nvvl_loader.cpp)

find_package(PkgConfig REQUIRED)
foreach(m avformat avcodec avfilter avutil)
    string(TOUPPER ${m} M)
    pkg_check_modules(${m} REQUIRED IMPORTED_TARGET lib${m})
    list(APPEND FFmpeg_LIBS PkgConfig::${m})
endforeach(m)

set(CMAKE_REQUIRED_INCLUDES ${avformat_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${avformat_LIBRARIES})
check_struct_has_member("struct AVStream" codecpar libavformat/avformat.h HAVE_AVSTREAM_CODECPAR LANGUAGE C)
set(CMAKE_EXTRA_INCLUDE_FILES libavcodec/avcodec.h)
check_type_size("AVBSFContext" AVBSFCONTEXT LANGUAGE CXX)

target_link_libraries(nvvl_loader PRIVATE cudart ${FFmpeg_LIBS})

target_link_libraries(nvvl_loader PUBLIC NVVL::NVVL)

if (WITH_OPENCV)
    # If we are in here, they explicitly asked for it, so make it required
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core cudaarithm cudaimgproc imgcodecs)
    target_compile_definitions(nvvl_loader PRIVATE HAVE_OPENCV=1)
    target_link_libraries(nvvl_loader PUBLIC opencv_core PRIVATE ${OpenCV_LIBS})
endif()

add_custom_target(examples DEPENDS nvvl_loader)
