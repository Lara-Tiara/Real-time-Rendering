#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <array>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <stb_image.h>

struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
};

struct GPUMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    glm::mat4 baseModel = glm::mat4(1.0f);
};

struct FloorVertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

static GLuint createCheckerTexture(int size, int cell)
{
    std::vector<unsigned char> img(size * size * 4);

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int cx = x / cell;
            int cy = y / cell;
            unsigned char v = ((cx + cy) & 1) ? 255 : 0;

            int i = (y * size + x) * 4;
            img[i + 0] = v;
            img[i + 1] = v;
            img[i + 2] = v;
            img[i + 3] = 255;
        }
    }

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.data());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenerateMipmap(GL_TEXTURE_2D);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static void applySamplerParams2D(GLuint tex, GLenum minF, GLenum magF)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)minF);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)magF);
    glBindTexture(GL_TEXTURE_2D, 0);
}

static void createFloorPlane(float halfSize, float y, float uvTiling,
                             GLuint& outVAO, GLuint& outVBO, GLuint& outEBO, GLsizei& outIndexCount)
{
    FloorVertex v[4] = {
        {-halfSize, y, -halfSize,  0.f, 1.f, 0.f,  0.f,       0.f},
        { halfSize, y, -halfSize,  0.f, 1.f, 0.f,  uvTiling,  0.f},
        { halfSize, y,  halfSize,  0.f, 1.f, 0.f,  uvTiling,  uvTiling},
        {-halfSize, y,  halfSize,  0.f, 1.f, 0.f,  0.f,       uvTiling}
    };

    unsigned int idx[6] = { 0, 1, 2, 0, 2, 3 };
    outIndexCount = 6;

    glGenVertexArrays(1, &outVAO);
    glGenBuffers(1, &outVBO);
    glGenBuffers(1, &outEBO);

    glBindVertexArray(outVAO);

    glBindBuffer(GL_ARRAY_BUFFER, outVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, outEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(FloorVertex), (void*)offsetof(FloorVertex, px));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(FloorVertex), (void*)offsetof(FloorVertex, nx));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(FloorVertex), (void*)offsetof(FloorVertex, u));

    glBindVertexArray(0);
}

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
        aiProcess_GenSmoothNormals |
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

        for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
            glm::vec3 p = aiToGlm(mesh->mVertices[i]);
            glm::vec3 n = mesh->HasNormals() ? glm::normalize(aiToGlm(mesh->mNormals[i])) : glm::vec3(0, 1, 0);

            outMin = glm::min(outMin, p);
            outMax = glm::max(outMax, p);

            outVertices.push_back(Vertex{p.x, p.y, p.z, n.x, n.y, n.z});
        }

        for (unsigned f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            for (unsigned j = 0; j < face.mNumIndices; ++j) {
                outIndices.push_back(baseVertex + face.mIndices[j]);
            }
        }

        baseVertex += mesh->mNumVertices;
    }
}

static glm::mat4 computeNormalizationModel(const glm::vec3& bmin, const glm::vec3& bmax, float targetSize = 1.4f) {
    glm::vec3 center = 0.5f * (bmin + bmax);
    glm::vec3 ext = (bmax - bmin);
    float maxExtent = std::max(ext.x, std::max(ext.y, ext.z));
    float scale = (maxExtent > 0.0f) ? (targetSize / maxExtent) : 1.0f;

    glm::mat4 M(1.0f);
    M = glm::scale(M, glm::vec3(scale));
    M = glm::translate(M, -center);
    return M;
}

static GPUMesh uploadMeshToGPU(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices, const glm::mat4& baseModel) {
    GPUMesh mesh;
    mesh.baseModel = baseModel;
    mesh.indexCount = (GLsizei)indices.size();

    glGenVertexArrays(1, &mesh.vao);
    glGenBuffers(1, &mesh.vbo);
    glGenBuffers(1, &mesh.ebo);

    glBindVertexArray(mesh.vao);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)(indices.size() * sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, px));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nx));

    glBindVertexArray(0);
    return mesh;
}

static void destroyMesh(GPUMesh& m) {
    if (m.ebo) glDeleteBuffers(1, &m.ebo);
    if (m.vbo) glDeleteBuffers(1, &m.vbo);
    if (m.vao) glDeleteVertexArrays(1, &m.vao);
    m = GPUMesh{};
}

static GLuint loadCubemap(const std::array<std::string, 6>& faces)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    stbi_set_flip_vertically_on_load(false);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    int baseW = -1, baseH = -1;

    for (GLuint i = 0; i < 6; ++i)
    {
        int w, h, n;
        unsigned char* data = stbi_load(faces[i].c_str(), &w, &h, &n, 3);

        if (!data) {
            std::cerr << "[Cubemap] FAIL face " << i << " : " << faces[i] << "\n"
                      << "  reason: " << (stbi_failure_reason() ? stbi_failure_reason() : "(unknown)") << "\n";
            throw std::runtime_error("Cubemap face failed to load.");
        }

        if (i == 0) { baseW = w; baseH = h; }
        if (w != baseW || h != baseH) {
            stbi_image_free(data);
            throw std::runtime_error("Cubemap faces must all be the same resolution.");
        }

        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                     0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

        stbi_image_free(data);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "[Cubemap] glTexImage2D error face " << i << " : 0x"
                      << std::hex << err << std::dec << "\n";
            throw std::runtime_error("OpenGL error uploading cubemap face.");
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    return tex;
}

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1400, 720, "Assignment 4 - Mip Mapping", nullptr, nullptr);
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
    glDepthFunc(GL_LESS);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    const std::string unicornPath = std::string(ASSET_DIR) + "/unicorn.glb";
    const std::string horsePath   = std::string(ASSET_DIR) + "/horse/source/poly.glb";
    const std::string moonPath    = std::string(ASSET_DIR) + "/moon/source/Moon.obj";

    auto loadOne = [&](const std::string& path, GPUMesh& outMesh) {
        std::vector<Vertex> v;
        std::vector<unsigned int> idx;
        glm::vec3 bmin, bmax;
        loadMergedMeshesAssimp(path, v, idx, bmin, bmax);
        std::cout << "Loaded " << path << " | vertices=" << v.size() << " | indices=" << idx.size() << "\n";
        glm::mat4 base = computeNormalizationModel(bmin, bmax, 1.4f);
        outMesh = uploadMeshToGPU(v, idx, base);
    };

    GPUMesh unicornMesh, horseMesh, moonMesh;
    try {
        loadOne(unicornPath, unicornMesh);
        loadOne(horsePath,   horseMesh);
        loadOne(moonPath,    moonMesh);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        destroyMesh(unicornMesh);
        destroyMesh(horseMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    float skyboxVerts[] = {
        -1.f,  1.f, -1.f,  -1.f, -1.f, -1.f,   1.f, -1.f, -1.f,
         1.f, -1.f, -1.f,   1.f,  1.f, -1.f,  -1.f,  1.f, -1.f,
        -1.f, -1.f,  1.f,  -1.f, -1.f, -1.f,  -1.f,  1.f, -1.f,
        -1.f,  1.f, -1.f,  -1.f,  1.f,  1.f,  -1.f, -1.f,  1.f,
         1.f, -1.f, -1.f,   1.f, -1.f,  1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,   1.f,  1.f, -1.f,   1.f, -1.f, -1.f,
        -1.f, -1.f,  1.f,  -1.f,  1.f,  1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,   1.f, -1.f,  1.f,  -1.f, -1.f,  1.f,
        -1.f,  1.f, -1.f,   1.f,  1.f, -1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,  -1.f,  1.f,  1.f,  -1.f,  1.f, -1.f,
        -1.f, -1.f, -1.f,  -1.f, -1.f,  1.f,   1.f, -1.f,  1.f,
         1.f, -1.f,  1.f,   1.f, -1.f, -1.f,  -1.f, -1.f, -1.f
    };

    GLuint skyVAO = 0, skyVBO = 0;
    glGenVertexArrays(1, &skyVAO);
    glGenBuffers(1, &skyVBO);
    glBindVertexArray(skyVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVerts), skyboxVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    const std::string skyDir = std::string(ASSET_DIR) + "/skybox";
    std::array<std::string, 6> faces = {
        skyDir + "/right.jpg",
        skyDir + "/left.jpg",
        skyDir + "/top.jpg",
        skyDir + "/bottom.jpg",
        skyDir + "/front.jpg",
        skyDir + "/back.jpg"
    };

    GLuint cubemapTex = 0;
    try {
        cubemapTex = loadCubemap(faces);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        destroyMesh(unicornMesh);
        destroyMesh(horseMesh);
        destroyMesh(moonMesh);
        glDeleteBuffers(1, &skyVBO);
        glDeleteVertexArrays(1, &skyVAO);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const char* modelVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vPosW;
out vec3 vNrmW;

void main(){
    vec4 posW = uModel * vec4(aPos, 1.0);
    vPosW = posW.xyz;

    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vNrmW = normalize(normalMat * aNrm);

    gl_Position = uProj * uView * posW;
}
)";

    const char* modelFS = R"(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;

out vec4 FragColor;

uniform vec3 uCamPosW;
uniform vec3 uLightDirW;
uniform vec3 uTint;

void main(){
    vec3 N = normalize(vNrmW);
    vec3 L = normalize(uLightDirW);
    vec3 V = normalize(uCamPosW - vPosW);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = uTint * NdotL;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 ambient = 0.10 * uTint;
    vec3 col = ambient + diffuse + 0.15 * spec;

    FragColor = vec4(col, 1.0);
}
)";

    const char* skyVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;

out vec3 vDir;

uniform mat4 uView;
uniform mat4 uProj;

void main() {
    vDir = aPos;
    vec4 p = uProj * uView * vec4(aPos, 1.0);
    gl_Position = p.xyww;
}
)";

    const char* skyFS = R"(
#version 330 core
in vec3 vDir;
out vec4 FragColor;

uniform samplerCube uEnvMap;

void main() {
    vec3 c = texture(uEnvMap, vDir).rgb;
    c = pow(max(c, vec3(0.0)), vec3(1.0/2.2));
    FragColor = vec4(c, 1.0);
}
)";

    const char* CheckerVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uUvTiling;

out vec3 vPosW;
out vec3 vNrmW;
out vec2 vUV;

void main(){
    vec4 posW = uModel * vec4(aPos, 1.0);
    vPosW = posW.xyz;

    mat3 normalMat = transpose(inverse(mat3(uModel)));
    vNrmW = normalize(normalMat * aNrm);

    vUV = aUV * uUvTiling;

    gl_Position = uProj * uView * posW;
}
)";

    const char* CheckerFS = R"(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;
in vec2 vUV;

out vec4 FragColor;

uniform vec3 uCamPosW;
uniform vec3 uLightDirW;

uniform sampler2D uFloorTex;

void main(){
    vec3 texCol = texture(uFloorTex, vUV).rgb;

    vec3 N = normalize(vNrmW);
    vec3 L = normalize(uLightDirW);
    vec3 V = normalize(uCamPosW - vPosW);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = texCol * NdotL;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 ambient = 0.08 * texCol;
    vec3 col = ambient + diffuse + 0.10 * spec;

    FragColor = vec4(col, 1.0);
}
)";

    GLuint modelProg = 0, skyProg = 0;
    try {
        GLuint vs = compileShader(GL_VERTEX_SHADER, modelVS);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, modelFS);
        modelProg = linkProgram(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);

        GLuint svs = compileShader(GL_VERTEX_SHADER, skyVS);
        GLuint sfs = compileShader(GL_FRAGMENT_SHADER, skyFS);
        skyProg = linkProgram(svs, sfs);
        glDeleteShader(svs);
        glDeleteShader(sfs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        glDeleteTextures(1, &cubemapTex);
        glDeleteBuffers(1, &skyVBO);
        glDeleteVertexArrays(1, &skyVAO);
        glDeleteProgram(modelProg);
        glDeleteProgram(skyProg);
        destroyMesh(unicornMesh);
        destroyMesh(horseMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const GLint m_uModel  = glGetUniformLocation(modelProg, "uModel");
    const GLint m_uView   = glGetUniformLocation(modelProg, "uView");
    const GLint m_uProj   = glGetUniformLocation(modelProg, "uProj");
    const GLint m_uCamPos = glGetUniformLocation(modelProg, "uCamPosW");
    const GLint m_uLight  = glGetUniformLocation(modelProg, "uLightDirW");
    const GLint m_uTint   = glGetUniformLocation(modelProg, "uTint");

    const GLint s_uView   = glGetUniformLocation(skyProg, "uView");
    const GLint s_uProj   = glGetUniformLocation(skyProg, "uProj");
    const GLint s_uEnvMap = glGetUniformLocation(skyProg, "uEnvMap");

    glUseProgram(skyProg);
    glUniform1i(s_uEnvMap, 0);

    GLuint floorVAO = 0, floorVBO = 0, floorEBO = 0;
    GLsizei floorIndexCount = 0;
    createFloorPlane(60.0f, -1.0f, 1.0f, floorVAO, floorVBO, floorEBO, floorIndexCount);

    GLuint floorProg = 0;
    {
        GLuint vs = compileShader(GL_VERTEX_SHADER, CheckerVS);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, CheckerFS);
        floorProg = linkProgram(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    const GLint f_uModel  = glGetUniformLocation(floorProg, "uModel");
    const GLint f_uView   = glGetUniformLocation(floorProg, "uView");
    const GLint f_uProj   = glGetUniformLocation(floorProg, "uProj");
    const GLint f_uCamPos = glGetUniformLocation(floorProg, "uCamPosW");
    const GLint f_uLight  = glGetUniformLocation(floorProg, "uLightDirW");
    const GLint f_uTex    = glGetUniformLocation(floorProg, "uFloorTex");
    const GLint f_uTiling = glGetUniformLocation(floorProg, "uUvTiling");

    glUseProgram(floorProg);
    glUniform1i(f_uTex, 0);

    static const char* kMinNames[6] = {
        "Nearest",
        "Linear",
        "Nearest Mipmap Nearest",
        "Linear  Mipmap Nearest",
        "Nearest Mipmap Linear",
        "Linear  Mipmap Linear"
    };
    static const GLenum kMinEnums[6] = {
        GL_NEAREST,
        GL_LINEAR,
        GL_NEAREST_MIPMAP_NEAREST,
        GL_LINEAR_MIPMAP_NEAREST,
        GL_NEAREST_MIPMAP_LINEAR,
        GL_LINEAR_MIPMAP_LINEAR
    };

    static const char* kMagNames[2] = { "Nearest", "Linear" };
    static const GLenum kMagEnums[2] = { GL_NEAREST, GL_LINEAR };

    static const int kResVals[] = { 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    static const char* kResNames[] = { "32", "64", "128", "256", "512", "1024", "2048", "4096" };
    static const int kResCount = 6;

    int checkerResIdx = 3;
    int checkerCellPx = 8;
    int minFilterIdx  = 5;
    int magFilterIdx  = 1;

    GLuint checkerTex = createCheckerTexture(kResVals[checkerResIdx], checkerCellPx);
    applySamplerParams2D(checkerTex, kMinEnums[minFilterIdx], kMagEnums[magFilterIdx]);

    float camYawDeg = 0.0f;
    float camPitchDeg = 10.0f;
    float camDist = 6.0f;

    bool showUnicorn = true;
    bool showMoon = true;
    bool showHorse = true;

    float rotUnicorn = 0.6f;
    float rotMoon    = 0.3f;
    float rotHorse   = 0.45f;

    float uvTiling = 120.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, 1);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Assignment 4 Controls");
        ImGui::Checkbox("Unicorn", &showUnicorn);
        ImGui::Checkbox("Moon", &showMoon);
        ImGui::Checkbox("Horse", &showHorse);
        ImGui::Separator();
        ImGui::SliderFloat("Cam Yaw (deg)", &camYawDeg, -180.0f, 180.0f);
        ImGui::SliderFloat("Cam Pitch (deg)", &camPitchDeg, -60.0f, 60.0f);
        ImGui::SliderFloat("Cam Distance", &camDist, 0.5f, 20.0f);
        ImGui::Separator();
        ImGui::SliderFloat("Unicorn Rot Speed", &rotUnicorn, -2.0f, 2.0f);
        ImGui::SliderFloat("Moon Rot Speed", &rotMoon, -2.0f, 2.0f);
        ImGui::SliderFloat("Horse Rot Speed", &rotHorse, -2.0f, 2.0f);

        ImGui::Separator();
        static const char* minItems[] = {
            "Nearest",
            "Linear",
            "Nearest Mipmap Nearest",
            "Linear Mipmap Nearest",
            "Nearest Mipmap Linear",
            "Linear Mipmap Linear"
        };
        static const char* magItems[] = {
            "Nearest",
            "Linear"
        };
        ImGui::Combo("Min Filter Mode", &minFilterIdx, minItems, 6);
        ImGui::Combo("Mag Filter Mode", &magFilterIdx, magItems, 2);
        ImGui::SliderFloat("UV Tiling", &uvTiling, 1.0f, 300.0f);

        static int lastResIdx = checkerResIdx;
        static int lastCellPx = checkerCellPx;
        static int lastMinIdx = minFilterIdx;
        static int lastMagIdx = magFilterIdx;

        ImGui::Separator();
        ImGui::Text("Checker Texture");

        ImGui::Combo("Checker Resolution", &checkerResIdx, kResNames, kResCount);
        ImGui::SliderInt("Checker Cell (px)", &checkerCellPx, 1, 128);
        ImGui::End();

        bool needRebuildChecker = (checkerResIdx != lastResIdx) || (checkerCellPx != lastCellPx);
        bool needUpdateSampler  = (minFilterIdx != lastMinIdx) || (magFilterIdx != lastMagIdx);

        if (needRebuildChecker) {
            if (checkerTex) glDeleteTextures(1, &checkerTex);
            checkerTex = createCheckerTexture(kResVals[checkerResIdx], checkerCellPx);

            applySamplerParams2D(checkerTex, kMinEnums[minFilterIdx], kMagEnums[magFilterIdx]);

            lastResIdx = checkerResIdx;
            lastCellPx = checkerCellPx;

            lastMinIdx = minFilterIdx;
            lastMagIdx = magFilterIdx;
        }
        else if (needUpdateSampler) {
            applySamplerParams2D(checkerTex, kMinEnums[minFilterIdx], kMagEnums[magFilterIdx]);
            lastMinIdx = minFilterIdx;
            lastMagIdx = magFilterIdx;
        }

        static int prevMin = -1;
        static int prevMag = -1;

        if (minFilterIdx != prevMin || magFilterIdx != prevMag) {
            static const GLenum minEnums[6] = {
                GL_NEAREST,
                GL_LINEAR,
                GL_NEAREST_MIPMAP_NEAREST,
                GL_LINEAR_MIPMAP_NEAREST,
                GL_NEAREST_MIPMAP_LINEAR,
                GL_LINEAR_MIPMAP_LINEAR
            };
            static const GLenum magEnums[2] = {
                GL_NEAREST,
                GL_LINEAR
            };

            glBindTexture(GL_TEXTURE_2D, checkerTex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)minEnums[minFilterIdx]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)magEnums[magFilterIdx]);

            if (minFilterIdx >= 2) {
                glGenerateMipmap(GL_TEXTURE_2D);
            }

            glBindTexture(GL_TEXTURE_2D, 0);

            prevMin = minFilterIdx;
            prevMag = magFilterIdx;
        }

        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        if (w <= 0 || h <= 0) {
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
            continue;
        }

        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float yaw = glm::radians(camYawDeg);
        float pitch = glm::radians(camPitchDeg);

        glm::vec3 target(0.0f, 0.2f, 0.0f);
        glm::vec3 camPos(
            target.x + camDist * cosf(pitch) * sinf(yaw),
            target.y + camDist * sinf(pitch),
            target.z + camDist * cosf(pitch) * cosf(yaw)
        );

        glm::mat4 P = glm::perspective(glm::radians(60.0f), (float)w / (float)h, 0.1f, 200.0f);
        glm::mat4 V = glm::lookAt(camPos, target, glm::vec3(0, 1, 0));

        glm::vec3 lightDirW = glm::normalize(glm::vec3(0.6f, 1.0f, 0.3f));
        float t = (float)glfwGetTime();

        glUseProgram(floorProg);
        glUniformMatrix4fv(f_uView, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(f_uProj, 1, GL_FALSE, glm::value_ptr(P));
        glUniform3fv(f_uCamPos, 1, glm::value_ptr(camPos));
        glUniform3fv(f_uLight, 1, glm::value_ptr(lightDirW));

        glm::mat4 Mf(1.0f);
        glUniformMatrix4fv(f_uModel, 1, GL_FALSE, glm::value_ptr(Mf));
        glUniform1f(f_uTiling, uvTiling);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, checkerTex);

        glBindVertexArray(floorVAO);
        glDrawElements(GL_TRIANGLES, floorIndexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glUseProgram(modelProg);
        glUniformMatrix4fv(m_uView, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(m_uProj, 1, GL_FALSE, glm::value_ptr(P));
        glUniform3fv(m_uCamPos, 1, glm::value_ptr(camPos));
        glUniform3fv(m_uLight, 1, glm::value_ptr(lightDirW));

        if (showUnicorn) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(-2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, t * rotUnicorn, glm::vec3(0, 1, 0));
            M = M * unicornMesh.baseModel;

            glUniformMatrix4fv(m_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(m_uTint, 0.85f, 0.75f, 0.95f);

            glBindVertexArray(unicornMesh.vao);
            glDrawElements(GL_TRIANGLES, unicornMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        if (showMoon) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(0.0f, 0.0f, 0.0f));
            M = glm::rotate(M, t * rotMoon, glm::vec3(0, 1, 0));
            M = M * moonMesh.baseModel;

            glUniformMatrix4fv(m_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(m_uTint, 0.75f, 0.75f, 0.75f);

            glBindVertexArray(moonMesh.vao);
            glDrawElements(GL_TRIANGLES, moonMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        if (showHorse) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, -t * rotHorse, glm::vec3(0, 1, 0));
            M = M * horseMesh.baseModel;

            glUniformMatrix4fv(m_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(m_uTint, 0.80f, 0.70f, 0.60f);

            glBindVertexArray(horseMesh.vao);
            glDrawElements(GL_TRIANGLES, horseMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);

        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);

        glUseProgram(skyProg);
        glm::mat4 Vsky = glm::mat4(glm::mat3(V));
        glUniformMatrix4fv(s_uView, 1, GL_FALSE, glm::value_ptr(Vsky));
        glUniformMatrix4fv(s_uProj, 1, GL_FALSE, glm::value_ptr(P));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTex);

        glBindVertexArray(skyVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    glDeleteProgram(modelProg);
    glDeleteProgram(skyProg);

    if (floorProg) glDeleteProgram(floorProg);
    if (checkerTex) glDeleteTextures(1, &checkerTex);

    if (floorEBO) glDeleteBuffers(1, &floorEBO);
    if (floorVBO) glDeleteBuffers(1, &floorVBO);
    if (floorVAO) glDeleteVertexArrays(1, &floorVAO);

    glDeleteTextures(1, &cubemapTex);

    destroyMesh(unicornMesh);
    destroyMesh(horseMesh);
    destroyMesh(moonMesh);

    glDeleteBuffers(1, &skyVBO);
    glDeleteVertexArrays(1, &skyVAO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
