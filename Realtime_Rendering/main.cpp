// main.cpp
// Loads a teapot (glTF/GLB) with Assimp, renders it 3 times side-by-side using:
// 0 = Blinn-Phong, 1 = Toon, 2 = Oren-Nayar (no textures)

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cfloat>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
};

static void framebuffer_size_callback(GLFWwindow*, int w, int h) {
    glViewport(0, 0, w, h);
}

static GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(std::max(1, len));
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error:\n" << log.data() << "\n";
        throw std::runtime_error("Shader compilation failed");
    }
    return s;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);

    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(std::max(1, len));
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link error:\n" << log.data() << "\n";
        throw std::runtime_error("Program link failed");
    }
    return p;
}

static glm::vec3 aiToGlm(const aiVector3D& v) {
    return glm::vec3(v.x, v.y, v.z);
}

// Loads *all* meshes and merges them into one vertex/index buffer.
// Uses PreTransformVertices so node transforms are baked (good for static scenes).
static void loadMergedMeshesAssimp(
    const std::string& path,
    std::vector<Vertex>& outVertices,
    std::vector<unsigned int>& outIndices,
    glm::vec3& outMin,
    glm::vec3& outMax
) {
    Assimp::Importer importer;

    const unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_GenNormals |
        aiProcess_PreTransformVertices;

    const aiScene* scene = importer.ReadFile(path, flags);
    if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        throw std::runtime_error(std::string("Assimp error: ") + importer.GetErrorString());
    }
    if (scene->mNumMeshes == 0) {
        throw std::runtime_error("No meshes found in: " + path);
    }

    outVertices.clear();
    outIndices.clear();

    outMin = glm::vec3( FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);

    size_t totalVerts = 0, totalIdx = 0;
    for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
        totalVerts += scene->mMeshes[m]->mNumVertices;
        for (unsigned f = 0; f < scene->mMeshes[m]->mNumFaces; ++f)
            totalIdx += scene->mMeshes[m]->mFaces[f].mNumIndices;
    }
    outVertices.reserve(totalVerts);
    outIndices.reserve(totalIdx);

    unsigned int baseVertex = 0;
    for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        // vertices
        for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
            glm::vec3 p = aiToGlm(mesh->mVertices[i]);
            glm::vec3 n = mesh->HasNormals() ? glm::normalize(aiToGlm(mesh->mNormals[i]))
                                             : glm::vec3(0, 1, 0);

            outMin = glm::min(outMin, p);
            outMax = glm::max(outMax, p);

            outVertices.push_back(Vertex{p.x, p.y, p.z, n.x, n.y, n.z});
        }

        // indices
        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            for (unsigned j = 0; j < face.mNumIndices; ++j) {
                outIndices.push_back(baseVertex + face.mIndices[j]);
            }
        }

        baseVertex += mesh->mNumVertices;
    }
}

int main() {
    // ---- GLFW init ----
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // macOS OpenGL 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1100, 720, "3 Teapots: Blinn-Phong / Toon / Oren-Nayar", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // ---- Load model ----
    std::string modelPath = std::string(ASSET_DIR) + "/teapot.gltf";
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    glm::vec3 bmin, bmax;

    try {
        loadMergedMeshesAssimp(modelPath, vertices, indices, bmin, bmax);
        std::cout << "Loaded " << modelPath
                  << " | vertices=" << vertices.size()
                  << " | indices=" << indices.size() << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // ---- Normalize model (center + scale) ----
    glm::vec3 center = 0.5f * (bmin + bmax);
    glm::vec3 ext = (bmax - bmin);
    float maxExtent = std::max(ext.x, std::max(ext.y, ext.z));
    float scale = (maxExtent > 0.0f) ? (1.4f / maxExtent) : 1.0f;

    // Base model: move to origin + uniform scale
    // NOTE: order matters (right-multiply): M = S * T(-center)
    glm::mat4 baseModel(1.0f);
    baseModel = glm::scale(baseModel, glm::vec3(scale));
    baseModel = glm::translate(baseModel, -center);

    // ---- Upload to GPU ----
    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(vertices.size() * sizeof(Vertex)),
                 vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(indices.size() * sizeof(unsigned int)),
                 indices.data(),
                 GL_STATIC_DRAW);

    // layout(location=0): position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, px));
    // layout(location=1): normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nx));

    glBindVertexArray(0);

    // ---- Shaders ----
    const char* vsSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vPosW;
out vec3 vNrmW;

void main() {
    vec4 posW = uModel * vec4(aPos, 1.0);
    vPosW = posW.xyz;

    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vNrmW = normalize(normalMat * aNrm);

    gl_Position = uProj * uView * posW;
}
)";

    const char* fsSrc = R"(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;

out vec4 FragColor;

uniform int  uModelType;   // 0=BlinnPhong, 1=Toon, 2=OrenNayar

uniform vec3  uAlbedo;
uniform vec3  uLightPosW;
uniform vec3  uLightColor;
uniform vec3  uCamPosW;

uniform float uKs;
uniform float uShininess;
uniform float uRoughness;   // 0..1
uniform float uToonLevels;  // e.g. 3..6

float saturate(float x){ return clamp(x, 0.0, 1.0); }

vec3 blinnPhong(vec3 N, vec3 V, vec3 L) {
    float ndotl = max(dot(N, L), 0.0);
    vec3 diffuse = uAlbedo * ndotl;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), uShininess);
    vec3 specular = uKs * spec * vec3(1.0);

    return (diffuse + specular) * uLightColor;
}

vec3 toon(vec3 N, vec3 V, vec3 L) {
    float ndotl = max(dot(N, L), 0.0);

    float levels = max(uToonLevels, 1.0);
    float q = floor(ndotl * levels) / levels;      // banded diffuse

    vec3 H = normalize(L + V);
    float s = pow(max(dot(N, H), 0.0), uShininess);
    float specBand = step(0.5, s);                 // hard spec highlight

    vec3 diffuse = uAlbedo * q;
    vec3 specular = uKs * specBand * vec3(1.0);

    return (diffuse + specular) * uLightColor;
}

// Oren–Nayar diffuse only
vec3 orenNayar(vec3 N, vec3 V, vec3 L) {
    float ndotl = max(dot(N, L), 0.0);
    float ndotv = max(dot(N, V), 0.0);
    if (ndotl <= 0.0 || ndotv <= 0.0) return vec3(0.0);

    // Map roughness [0,1] to sigma in radians (simple course-friendly mapping)
    float sigma = uRoughness * 1.57079632679; // ~pi/2
    float sigma2 = sigma * sigma;

    float A = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    float theta_i = acos(clamp(ndotl, 0.0, 1.0));
    float theta_r = acos(clamp(ndotv, 0.0, 1.0));

    float alpha = max(theta_i, theta_r);
    float beta  = min(theta_i, theta_r);

    vec3 vPerpN = normalize(V - N * ndotv);
    vec3 lPerpN = normalize(L - N * ndotl);
    float cosPhi = dot(vPerpN, lPerpN);

    float oren = ndotl * (A + B * max(0.0, cosPhi) * sin(alpha) * tan(beta));
    return uAlbedo * oren * uLightColor;
}

void main() {
    vec3 N = normalize(vNrmW);
    vec3 V = normalize(uCamPosW - vPosW);
    vec3 L = normalize(uLightPosW - vPosW);

    vec3 c;
    if (uModelType == 0)      c = blinnPhong(N, V, L);
    else if (uModelType == 1) c = toon(N, V, L);
    else                      c = orenNayar(N, V, L);

    vec3 ambient = 0.08 * uAlbedo * uLightColor;
    FragColor = vec4(ambient + c, 1.0);
}
)";

    GLuint vs = 0, fs = 0, prog = 0;
    try {
        vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        prog = linkProgram(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Cache uniform locations (avoid glGetUniformLocation every draw)
    const GLint loc_uModel      = glGetUniformLocation(prog, "uModel");
    const GLint loc_uView       = glGetUniformLocation(prog, "uView");
    const GLint loc_uProj       = glGetUniformLocation(prog, "uProj");
    const GLint loc_uModelType  = glGetUniformLocation(prog, "uModelType");

    const GLint loc_uAlbedo     = glGetUniformLocation(prog, "uAlbedo");
    const GLint loc_uLightPosW  = glGetUniformLocation(prog, "uLightPosW");
    const GLint loc_uLightColor = glGetUniformLocation(prog, "uLightColor");
    const GLint loc_uCamPosW    = glGetUniformLocation(prog, "uCamPosW");

    const GLint loc_uKs         = glGetUniformLocation(prog, "uKs");
    const GLint loc_uShininess  = glGetUniformLocation(prog, "uShininess");
    const GLint loc_uRoughness  = glGetUniformLocation(prog, "uRoughness");
    const GLint loc_uToonLevels = glGetUniformLocation(prog, "uToonLevels");

    // Scene constants
    const glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    const glm::vec3 albedo(0.75f, 0.78f, 0.85f);

    // Side-by-side positions
    const float offsets[3] = {-1.6f, 0.0f, 1.6f};

    // ---- Render loop ----
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);

        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        if (w <= 0 || h <= 0) continue;

        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera (fixed, looking at center teapot)
        glm::vec3 camPos(0.0f, 1.1f, 4.0f);
        glm::mat4 P = glm::perspective(glm::radians(60.0f), (float)w / (float)h, 0.1f, 100.0f);
        glm::mat4 V = glm::lookAt(camPos, glm::vec3(0.0f, 0.3f, 0.0f), glm::vec3(0, 1, 0));

        // Light position (world)
        glm::vec3 lightPos(2.5f, 3.0f, 2.0f);

        glUseProgram(prog);
        glUniformMatrix4fv(loc_uView, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(loc_uProj, 1, GL_FALSE, glm::value_ptr(P));

        glUniform3fv(loc_uLightPosW, 1, glm::value_ptr(lightPos));
        glUniform3fv(loc_uLightColor, 1, glm::value_ptr(lightColor));
        glUniform3fv(loc_uCamPosW, 1, glm::value_ptr(camPos));

        glUniform3fv(loc_uAlbedo, 1, glm::value_ptr(albedo));

        glBindVertexArray(vao);

        float t = (float)glfwGetTime();

        for (int i = 0; i < 3; ++i) {
            int modelType = i; // 0=Blinn-Phong, 1=Toon, 2=Oren-Nayar
            glUniform1i(loc_uModelType, modelType);

            glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(offsets[i], 0.0f, 0.0f));
            const float omega = 0.8f; // radians/sec (tweak if you want faster/slower)
            glm::mat4 R = glm::rotate(glm::mat4(1.0f), t * omega, glm::vec3(0, 1, 0));
            glm::mat4 M = T * R * baseModel;

            glUniformMatrix4fv(loc_uModel, 1, GL_FALSE, glm::value_ptr(M));

            // Model-specific parameters (tweak as you like)
            if (modelType == 0) {            // Blinn-Phong
                glUniform1f(loc_uKs, 0.55f);
                glUniform1f(loc_uShininess, 96.0f);
                glUniform1f(loc_uToonLevels, 4.0f); // unused
                glUniform1f(loc_uRoughness, 0.5f);  // unused
            } else if (modelType == 1) {     // Toon (NPR)
                glUniform1f(loc_uKs, 0.25f);
                glUniform1f(loc_uShininess, 48.0f);
                glUniform1f(loc_uToonLevels, 4.0f);
                glUniform1f(loc_uRoughness, 0.5f);  // unused
            } else {                          // Oren-Nayar (rough diffuse)
                glUniform1f(loc_uKs, 0.0f);          // diffuse-only here
                glUniform1f(loc_uShininess, 1.0f);   // unused
                glUniform1f(loc_uToonLevels, 4.0f);  // unused
                glUniform1f(loc_uRoughness, 0.75f);  // 0..1 (higher = rougher)
            }

            glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);
        glfwSwapBuffers(window);
    }

    // ---- Cleanup ----
    glDeleteProgram(prog);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
