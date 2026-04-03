cmake_minimum_required(VERSION 3.21)
project(OpenGL_CLion LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
        assimp
        GIT_REPOSITORY https://github.com/assimp/assimp.git
        GIT_TAG        v6.0.2
)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(assimp)

find_package(OpenGL REQUIRED)

add_executable(app
        main.cpp
        external/glad/src/glad.c
        src/renderer/Shader.cpp
        src/renderer/Shader.h
        src/renderer/Texture.cpp
        src/renderer/Texture.h
        src/renderer/Mesh.cpp
        src/renderer/Mesh.h
        src/renderer/Model.cpp
        src/renderer/Model.h
        src/material/ClothMaterial.cpp
        src/material/ClothMaterial.h
        src/material/MaterialPresets.cpp
        src/material/MaterialPresets.h
        src/ui/ClothViewerPanel.cpp
        src/ui/ClothViewerPanel.h
        src/AppState.cpp
        src/AppState.h
)

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)

target_sources(app PRIVATE
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(app PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        external/glad/include
)

target_compile_definitions(app PRIVATE
        IMGUI_IMPL_OPENGL_LOADER_GLAD
        ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)

if(APPLE)
    target_compile_definitions(app PRIVATE GL_SILENCE_DEPRECATION)
endif()

target_link_libraries(app PRIVATE
        glfw
        glm::glm
        assimp::assimp
        OpenGL::GL
)