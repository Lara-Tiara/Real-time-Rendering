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
    float u, v;
    float tx, ty, tz, tw;
};


struct GPUMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    glm::mat4 baseModel = glm::mat4(1.0f);
};

struct TextureSet {
    std::string name;
    GLuint albedo = 0;
    GLuint normal = 0;
    GLuint height = 0;
};

static void bindTextureSet(const TextureSet& ts) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ts.albedo);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, ts.normal);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, ts.height);
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
        aiProcess_CalcTangentSpace |
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
        for (unsigned f = 0; f < scene->mMeshes[m]->mNumFaces; ++f) {
            totalIdx += scene->mMeshes[m]->mFaces[f].mNumIndices;
        }
    }
    outVertices.reserve(totalVerts);
    outIndices.reserve(totalIdx);

    unsigned int baseVertex = 0;
    for (unsigned m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];

        for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
            glm::vec3 p = aiToGlm(mesh->mVertices[i]);
            glm::vec3 n = mesh->HasNormals() ? glm::normalize(aiToGlm(mesh->mNormals[i])) : glm::vec3(0, 1, 0);

            glm::vec2 uv(0.0f);
            if (mesh->HasTextureCoords(0)) {
                uv = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            }

            glm::vec3 T(1.0f, 0.0f, 0.0f);
            float sign = 1.0f;

            if (mesh->HasTangentsAndBitangents()) {
                T = aiToGlm(mesh->mTangents[i]);
                glm::vec3 B = aiToGlm(mesh->mBitangents[i]);

                T = glm::normalize(T - n * glm::dot(n, T));
                sign = (glm::dot(glm::cross(n, T), B) < 0.0f) ? -1.0f : 1.0f;
            } else {
                if (fabsf(n.x) > 0.9f) T = glm::vec3(0, 0, 1);
                T = glm::normalize(T - n * glm::dot(n, T));
                sign = 1.0f;
            }

            outMin = glm::min(outMin, p);
            outMax = glm::max(outMax, p);

            outVertices.push_back(Vertex{
                p.x, p.y, p.z,
                n.x, n.y, n.z,
                uv.x, uv.y,
                T.x, T.y, T.z, sign
            });
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

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, u));

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tx));

    glBindVertexArray(0);
    return mesh;
}

static void destroyMesh(GPUMesh& m) {
    if (m.ebo) glDeleteBuffers(1, &m.ebo);
    if (m.vbo) glDeleteBuffers(1, &m.vbo);
    if (m.vao) glDeleteVertexArrays(1, &m.vao);
    m = GPUMesh{};
}

static GLuint loadTexture2D(const std::string& path, bool flipY) {
    stbi_set_flip_vertically_on_load(flipY ? 1 : 0);

    int w = 0, h = 0, comp = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &comp, 0);
    if (!data) {
        throw std::runtime_error("Failed to load texture: " + path);
    }

    GLenum format = GL_RGB;
    if (comp == 1) format = GL_RED;
    else if (comp == 3) format = GL_RGB;
    else if (comp == 4) format = GL_RGBA;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

static GLuint make1x1RedTexture(unsigned char r) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, &r);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glBindTexture(GL_TEXTURE_2D, 0);
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

    GLFWwindow* window = glfwCreateWindow(1400, 720, "Realtime Rendering Assignment 3", nullptr, nullptr);
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

    GLuint defaultHeightTex = make1x1RedTexture(128);

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

    const char* basicVS = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUV;
layout(location=3) in vec4 aTan;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vPosW;
out vec2 vUV;
out mat3 vTBN;

void main(){
    vec4 posW = uModel * vec4(aPos, 1.0);
    vPosW = posW.xyz;
    vUV = aUV;

    mat3 normalMat = transpose(inverse(mat3(uModel)));

    vec3 N = normalize(normalMat * aNrm);
    vec3 T = normalize(normalMat * aTan.xyz);
    T = normalize(T - N * dot(N, T));
    vec3 B = aTan.w * normalize(cross(N, T));

    vTBN = mat3(T, B, N);
    gl_Position = uProj * uView * posW;
}
)";


    const char* basicFS = R"(
#version 330 core
in vec3 vPosW;
in vec2 vUV;
in mat3 vTBN;

out vec4 FragColor;

uniform vec3 uCamPosW;
uniform vec3 uLightDirW;
uniform vec3 uAlbedoTint;

uniform sampler2D uAlbedoTex;
uniform sampler2D uNormalTex;
uniform sampler2D uHeightTex;

uniform int   uUseNormalMap;
uniform int   uUseBumpMap;

uniform float uNormalStrength;
uniform float uBumpStrength;

vec3 bumpNormalFromHeight()
{
    ivec2 ts = textureSize(uHeightTex, 0);
    vec2 texel = 1.0 / vec2(max(ts, ivec2(1)));

    float hL = texture(uHeightTex, vUV - vec2(texel.x, 0.0)).r;
    float hR = texture(uHeightTex, vUV + vec2(texel.x, 0.0)).r;
    float hD = texture(uHeightTex, vUV - vec2(0.0, texel.y)).r;
    float hU = texture(uHeightTex, vUV + vec2(0.0, texel.y)).r;

    vec2 g = vec2(hL - hR, hD - hU) * uBumpStrength;
    return normalize(vec3(g.x, g.y, 1.0));
}

void main(){
    vec3 albedo = texture(uAlbedoTex, vUV).rgb * uAlbedoTint;

    vec3 nTan = vec3(0.0, 0.0, 1.0);

    if (uUseNormalMap != 0) {
        vec3 nTex = texture(uNormalTex, vUV).rgb;
        vec3 nMap = normalize(nTex * 2.0 - 1.0);
        nMap = normalize(mix(vec3(0.0, 0.0, 1.0), nMap, clamp(uNormalStrength, 0.0, 1.0)));
        nTan = nMap;
    }

    if (uUseBumpMap != 0) {
        vec3 nBump = bumpNormalFromHeight();
        nTan = (uUseNormalMap != 0)
            ? normalize(vec3(nTan.xy + nBump.xy, nTan.z))
            : nBump;
    }

    vec3 N = normalize(vTBN * nTan);
    vec3 L = normalize(uLightDirW);
    vec3 V = normalize(uCamPosW - vPosW);

    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * NdotL;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0);

    vec3 ambient = 0.08 * albedo;
    vec3 col = ambient + diffuse + 0.15 * spec;

    FragColor = vec4(col, 1.0);
}
)";


    GLuint basicProg = 0;
    try {
        GLuint vs = compileShader(GL_VERTEX_SHADER, basicVS);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, basicFS);
        basicProg = linkProgram(vs, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        destroyMesh(unicornMesh);
        destroyMesh(horseMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const GLint b_uModel  = glGetUniformLocation(basicProg, "uModel");
    const GLint b_uView   = glGetUniformLocation(basicProg, "uView");
    const GLint b_uProj   = glGetUniformLocation(basicProg, "uProj");
    const GLint b_uCamPos = glGetUniformLocation(basicProg, "uCamPosW");
    const GLint b_uLight  = glGetUniformLocation(basicProg, "uLightDirW");

    const GLint b_uAlbedoTint = glGetUniformLocation(basicProg, "uAlbedoTint");
    const GLint b_uAlbedoTex  = glGetUniformLocation(basicProg, "uAlbedoTex");
    const GLint b_uNormalTex  = glGetUniformLocation(basicProg, "uNormalTex");
    const GLint b_uNormalStr  = glGetUniformLocation(basicProg, "uNormalStrength");

    const GLint b_uHeightTex   = glGetUniformLocation(basicProg, "uHeightTex");
    const GLint b_uUseNormal   = glGetUniformLocation(basicProg, "uUseNormalMap");
    const GLint b_uUseBump     = glGetUniformLocation(basicProg, "uUseBumpMap");
    const GLint b_uBumpStr     = glGetUniformLocation(basicProg, "uBumpStrength");

    float camYawDeg = 0.0f;
    float camPitchDeg = 10.0f;
    float camDist = 3.0f;

    bool showUnicorn = true;
    bool showMoon = true;
    bool showHorse = true;

    GLuint albedoTex = 0;
    GLuint normalTex = 0;

    std::vector<TextureSet> texSets;
    texSets.reserve(8);

    try {
        texSets.push_back(TextureSet{
            "Moon(Normal Mapping Only)",
            loadTexture2D(std::string(ASSET_DIR) + "/moon/textures/lroc_color_poles_1k.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/moon/textures/ldisplacement.jpg", true),
            defaultHeightTex
        });

        texSets.push_back(TextureSet{
            "BronzeGrid-3",
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_03_basecolor.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_03_normal.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_03_height.jpg", true)
        });

        texSets.push_back(TextureSet{
            "BronzeGrid-1",
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_01_basecolor.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_01_normal.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeGrid_01_height.jpg", true)
        });

        texSets.push_back(TextureSet{
            "BronzeGrid-2",
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeTiles_01_basecolor.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeTiles_01_normal.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/BronzeTiles_01_height.jpg", true)
        });

        texSets.push_back(TextureSet{
            "MetalTiles",
            loadTexture2D(std::string(ASSET_DIR) + "/textures/MetalTiles_01_basecolor.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/MetalTiles_01_normal.jpg", true),
            loadTexture2D(std::string(ASSET_DIR) + "/textures/MetalTiles_01_height.jpg", true)
        });


    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        glDeleteProgram(basicProg);
        destroyMesh(unicornMesh);
        destroyMesh(horseMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    std::vector<const char*> texSetNames;
    texSetNames.reserve(texSets.size());
    for (auto& s : texSets) texSetNames.push_back(s.name.c_str());

    glUseProgram(basicProg);
    glUniform1i(b_uAlbedoTex, 0);
    glUniform1i(b_uNormalTex, 1);
    glUniform1i(b_uHeightTex, 2);


    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, 1);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        static bool useNormalMap = false;
        static bool useBumpMap = false;
        static float bumpStrength = 0.0f;
        static float normalStrength = 0.0f;
        static int activeSet = 0;

        ImGui::Begin("Assignment 3 Controls");
        ImGui::Checkbox("Unicorn", &showUnicorn);
        ImGui::Checkbox("Moon", &showMoon);
        ImGui::Checkbox("Horse", &showHorse);
        ImGui::Separator();
        ImGui::SliderFloat("Cam Yaw (deg)", &camYawDeg, -180.0f, 180.0f);
        ImGui::SliderFloat("Cam Pitch (deg)", &camPitchDeg, -60.0f, 60.0f);
        ImGui::SliderFloat("Cam Distance", &camDist, 0.1f, 10.0f);
        ImGui::Separator();
        ImGui::Checkbox("Bump Mapping", &useBumpMap);
        ImGui::Checkbox("Normal Mapping", &useNormalMap);
        ImGui::SliderFloat("Bump Strength", &bumpStrength, 0.0f, 20.0f);
        ImGui::SliderFloat("Normal Strength", &normalStrength, 0.0f, 1.0f);
        ImGui::Combo("Texture Set (All)", &activeSet, texSetNames.data(), (int)texSetNames.size());
        ImGui::End();

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

        glUseProgram(basicProg);

        glUniformMatrix4fv(b_uView, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(b_uProj, 1, GL_FALSE, glm::value_ptr(P));
        glUniform3fv(b_uCamPos, 1, glm::value_ptr(camPos));
        glUniform3fv(b_uLight, 1, glm::value_ptr(lightDirW));

        bindTextureSet(texSets[activeSet]);
        glUniform1f(b_uNormalStr, normalStrength);

        glUniform1i(b_uUseNormal, useNormalMap ? 1 : 0);
        glUniform1i(b_uUseBump,   useBumpMap ? 1 : 0);
        glUniform1f(b_uBumpStr, bumpStrength);

        float t = (float)glfwGetTime();

        if (showUnicorn) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(-2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, t * 0.6f, glm::vec3(0, 1, 0));
            M = M * unicornMesh.baseModel;

            glUniformMatrix4fv(b_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(b_uAlbedoTint, 0.85f, 0.75f, 0.95f);

            glBindVertexArray(unicornMesh.vao);
            glDrawElements(GL_TRIANGLES, unicornMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        if (showMoon) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(0.0f, 0.0f, 0.0f));
            M = glm::rotate(M, t * 0.3f, glm::vec3(0, 1, 0));
            M = M * moonMesh.baseModel;

            glUniformMatrix4fv(b_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(b_uAlbedoTint, 0.75f, 0.75f, 0.75f);

            glBindVertexArray(moonMesh.vao);
            glDrawElements(GL_TRIANGLES, moonMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        if (showHorse) {
            glm::mat4 M(1.0f);
            M = glm::translate(M, glm::vec3(2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, -t * 0.45f, glm::vec3(0, 1, 0));
            M = M * horseMesh.baseModel;

            glUniformMatrix4fv(b_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glUniform3f(b_uAlbedoTint, 0.80f, 0.70f, 0.60f);

            glBindVertexArray(horseMesh.vao);
            glDrawElements(GL_TRIANGLES, horseMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    glDeleteProgram(basicProg);

    destroyMesh(unicornMesh);
    destroyMesh(horseMesh);
    destroyMesh(moonMesh);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    for (auto& s : texSets) {
        if (s.albedo) glDeleteTextures(1, &s.albedo);
        if (s.normal) glDeleteTextures(1, &s.normal);
        if (s.height && s.height != defaultHeightTex) glDeleteTextures(1, &s.height);
    }
    if (defaultHeightTex) glDeleteTextures(1, &defaultHeightTex);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
