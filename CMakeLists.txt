cmake_minimum_required(VERSION 3.8)
project(Physx-Test)
set(CMAKE_BUILD_TYPE RELEASE)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(include)

set(SRCS
    ./src/SnippetPBDCloth.cpp
)

set(outputName run)

set(RELEASE_PHYSX 
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXGpu_64.so
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysX_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXCharacterKinematic_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXCommon_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXCooking_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXExtensions_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXFoundation_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXPvdSDK_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXVehicle_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPhysXVehicle2_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libPVDRuntime_64.so
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libSnippetRender_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/release/libSnippetUtils_static_64.a
)

set(DEBUG_PHYSX
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXGpu_64.so
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysX_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXCharacterKinematic_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXCommon_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXCooking_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXExtensions_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXFoundation_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXPvdSDK_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXVehicle_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPhysXVehicle2_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libPVDRuntime_64.so
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libSnippetRender_static_64.a
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/debug/libSnippetUtils_static_64.a
)

add_executable(${outputName} ${SRCS})
target_include_directories(${outputName} PUBLIC ${OPENGL_INCLUDE_DIR})
target_include_directories(${outputName} PUBLIC ${GLUT_INCLUDE_DIR})
target_link_libraries(${outputName} PUBLIC glfw ${OPENGL_gl_LIBRARY})
target_link_libraries(${outputName} PUBLIC glut)
target_link_libraries(${outputName} PUBLIC GLU)
target_link_libraries(${outputName} PRIVATE CUDA::cudart_static)
target_link_libraries(${outputName} PRIVATE CUDA::cuda_driver)
if(${CMAKE_BUILD_TYPE} EQUAL "DEBUG")
    target_link_libraries(${outputName} PUBLIC ${DEBUG_PHYSX})
    add_compile_definitions(RENDER_SNIPPET)
    remove_definitions("NDEBUG")
else() 
    target_link_libraries(${outputName} PUBLIC ${RELEASE_PHYSX})
    
    add_compile_definitions(NDEBUG)
    add_compile_definitions(RENDER_SNIPPET)
endif()

