cmake_minimum_required(VERSION 3.21)
project(OpenGL_CLion_Mac LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

# ---- GLFW (build from source) ----
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.3.9
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ---- GLM (header-only) ----
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# ---- ASSIMP ----
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

# ---- Your executable ----
add_executable(app
        main.cpp
        external/glad/src/glad.c
)

target_include_directories(app PRIVATE
        external/glad/include
)

target_compile_definitions(app PRIVATE ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

# macOS OpenGL framework
find_library(OPENGL_LIBRARY OpenGL)

# Link everything here (after app exists)
target_link_libraries(app PRIVATE
        glfw
        glm::glm
        assimp::assimp
        ${OPENGL_LIBRARY}
)
