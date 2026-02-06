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

        for (unsigned i = 0; i < mesh->mNumVertices; ++i) {
            glm::vec3 p = aiToGlm(mesh->mVertices[i]);
            glm::vec3 n = mesh->HasNormals() ? glm::normalize(aiToGlm(mesh->mNormals[i]))
                                             : glm::vec3(0, 1, 0);

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
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(vertices.size() * sizeof(Vertex)),
                 vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(indices.size() * sizeof(unsigned int)),
                 indices.data(),
                 GL_STATIC_DRAW);

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
        unsigned char* data = stbi_load(faces[i].c_str(), &w, &h, &n, 3); // force RGB

        if (!data) {
            std::cerr << "[Cubemap] FAIL face " << i << " : " << faces[i] << "\n"
                      << "  reason: " << (stbi_failure_reason() ? stbi_failure_reason() : "(unknown)") << "\n";
            throw std::runtime_error("Cubemap face failed to load.");
        }

        std::cerr << "[Cubemap] OK   face " << i << " : " << faces[i]
                  << " (" << w << "x" << h << ", srcChannels=" << n << ", forced=3)\n";

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

    GLFWwindow* window = glfwCreateWindow(1400, 720, "Lab 2 - Transmittance (Fresnel + Dispersion + Cubemap)", nullptr, nullptr);
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

    const std::string testPath = std::string(ASSET_DIR) + "/test.obj";
    const std::string capPath  = std::string(ASSET_DIR) + "/cap.obj";
    const std::string moonPath = std::string(ASSET_DIR) + "/Moon.obj";

    auto loadOne = [&](const std::string& path, GPUMesh& outMesh) {
        std::vector<Vertex> v;
        std::vector<unsigned int> idx;
        glm::vec3 bmin, bmax;
        loadMergedMeshesAssimp(path, v, idx, bmin, bmax);
        std::cout << "Loaded " << path << " | vertices=" << v.size() << " | indices=" << idx.size() << "\n";
        glm::mat4 base = computeNormalizationModel(bmin, bmax, 1.4f);
        outMesh = uploadMeshToGPU(v, idx, base);
    };

    GPUMesh testMesh, capMesh, moonMesh, staMesh;
    try {
        loadOne(testPath, testMesh);
        loadOne(capPath,  capMesh);
        loadOne(moonPath, moonMesh);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        destroyMesh(testMesh);
        destroyMesh(capMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // ---- Skybox geometry (cube) ----
    float skyboxVerts[] = {
        // back
        -1.f,  1.f, -1.f,  -1.f, -1.f, -1.f,   1.f, -1.f, -1.f,
         1.f, -1.f, -1.f,   1.f,  1.f, -1.f,  -1.f,  1.f, -1.f,
        // left
        -1.f, -1.f,  1.f,  -1.f, -1.f, -1.f,  -1.f,  1.f, -1.f,
        -1.f,  1.f, -1.f,  -1.f,  1.f,  1.f,  -1.f, -1.f,  1.f,
        // right
         1.f, -1.f, -1.f,   1.f, -1.f,  1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,   1.f,  1.f, -1.f,   1.f, -1.f, -1.f,
        // front
        -1.f, -1.f,  1.f,  -1.f,  1.f,  1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,   1.f, -1.f,  1.f,  -1.f, -1.f,  1.f,
        // top
        -1.f,  1.f, -1.f,   1.f,  1.f, -1.f,   1.f,  1.f,  1.f,
         1.f,  1.f,  1.f,  -1.f,  1.f,  1.f,  -1.f,  1.f, -1.f,
        // bottom
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

    // ---- Load cubemap ----
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
        destroyMesh(testMesh);
        destroyMesh(capMesh);
        destroyMesh(moonMesh);
        destroyMesh(staMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // ---- Shaders ----
    const char* glassVS = R"(
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

    // UPDATED glassFS
    const char* glassFS = R"(
#version 330 core
in vec3 vPosW;
in vec3 vNrmW;

out vec4 FragColor;

uniform samplerCube uEnvMap;
uniform vec3  uCamPosW;

uniform float uIOR;              // e.g. glass ~1.5, diamond ~2.42
uniform float uDispersion;       // relative, try 0..0.2 (stylized)
uniform vec3  uTint;

uniform float uRefractStrength;  // 0..1 (blend I -> refracted dir)
uniform float uFresnelStrength;  // 0..1.5 (reduces/boosts reflection dominance)

float saturate(float x){ return clamp(x, 0.0, 1.0); }

vec3 srgbToLinear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 sampleEnvLinear(vec3 dir) {
    vec3 c = texture(uEnvMap, dir).rgb;  // LDR cubemap assumed sRGB-ish
    return srgbToLinear(c);              // convert to linear for mixing/lighting
}

void main() {
    vec3 N = normalize(vNrmW);
    vec3 V = normalize(uCamPosW - vPosW);   // frag -> camera
    if (dot(N, V) < 0.0) N = -N;            // IMPORTANT: keep normal facing viewer

    vec3 I = -V;                            // camera -> frag

    // Reflection (linear)
    vec3 R = reflect(I, N);
    vec3 refl = sampleEnvLinear(R);

    // Dispersion on eta (relative), not by adding/subtracting to IOR directly
    float eta  = 1.0 / uIOR;
    float etaR = eta * (1.0 - uDispersion);
    float etaG = eta;
    float etaB = eta * (1.0 + uDispersion);

    vec3 TrFull = refract(I, N, etaR);
    vec3 TgFull = refract(I, N, etaG);
    vec3 TbFull = refract(I, N, etaB);

    // Handle total internal reflection: refract returns (0,0,0)
    if (dot(TrFull, TrFull) < 1e-6) TrFull = R;
    if (dot(TgFull, TgFull) < 1e-6) TgFull = R;
    if (dot(TbFull, TbFull) < 1e-6) TbFull = R;

    // Artistic control: blend between no-bend (I) and full refract
    vec3 Tr = normalize(mix(I, TrFull, uRefractStrength));
    vec3 Tg = normalize(mix(I, TgFull, uRefractStrength));
    vec3 Tb = normalize(mix(I, TbFull, uRefractStrength));

    // Refraction sampling (linear), but take R/G/B from different directions
    vec3 refr;
    refr.r = sampleEnvLinear(Tr).r;
    refr.g = sampleEnvLinear(Tg).g;
    refr.b = sampleEnvLinear(Tb).b;

    refr *= uTint; // treat tint as linear multiplier

    // Fresnel (Schlick)
    float f0 = (uIOR - 1.0) / (uIOR + 1.0);
    f0 = f0 * f0;

    float cosTheta = saturate(dot(N, V));
    float F = f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);

    // Control how much reflection hides refraction/dispersion
    F = clamp(F * uFresnelStrength, 0.0, 1.0);

    // Mix in linear
    vec3 col = mix(refr, refl, F);

    // Simple tonemap + gamma for display
    col = col / (col + vec3(1.0));
    col = pow(max(col, vec3(0.0)), vec3(1.0/2.2));

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
    // keep as-is (your skybox is LDR JPG; this displays fine for the lab)
    vec3 c = texture(uEnvMap, vDir).rgb;
    c = pow(max(c, vec3(0.0)), vec3(1.0/2.2));
    FragColor = vec4(c, 1.0);
}
)";

    GLuint glassProg = 0, skyProg = 0;
    try {
        GLuint vs = compileShader(GL_VERTEX_SHADER, glassVS);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, glassFS);
        glassProg = linkProgram(vs, fs);
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
        destroyMesh(testMesh);
        destroyMesh(capMesh);
        destroyMesh(moonMesh);
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    const GLint g_uModel    = glGetUniformLocation(glassProg, "uModel");
    const GLint g_uView     = glGetUniformLocation(glassProg, "uView");
    const GLint g_uProj     = glGetUniformLocation(glassProg, "uProj");
    const GLint g_uEnvMap   = glGetUniformLocation(glassProg, "uEnvMap");
    const GLint g_uCamPosW  = glGetUniformLocation(glassProg, "uCamPosW");
    const GLint g_uIOR      = glGetUniformLocation(glassProg, "uIOR");
    const GLint g_uDisp     = glGetUniformLocation(glassProg, "uDispersion");
    const GLint g_uTint     = glGetUniformLocation(glassProg, "uTint");

    const GLint g_uRefractStrength = glGetUniformLocation(glassProg, "uRefractStrength");
    const GLint g_uFresnelStrength = glGetUniformLocation(glassProg, "uFresnelStrength");

    const GLint s_uView     = glGetUniformLocation(skyProg, "uView");
    const GLint s_uProj     = glGetUniformLocation(skyProg, "uProj");
    const GLint s_uEnvMap   = glGetUniformLocation(skyProg, "uEnvMap");

    glUseProgram(glassProg);
    glUniform1i(g_uEnvMap, 0);

    glUseProgram(skyProg);
    glUniform1i(s_uEnvMap, 0);

    // ---- UI parameters ----
    float ior = 1.50f;

    float dispersion = 0.05f;

    glm::vec3 tint(1.0f, 1.0f, 1.0f);

    float refrStrength = 0.65f;
    float fresnelStrength = 1.0f;

    float camYawDeg = 0.0f;
    float camPitchDeg = 10.0f;
    float camDist = 6.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);

        // ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Lab 2 Controls");
        ImGui::SliderFloat("IOR", &ior, 1.0f, 2.4f);

        ImGui::SliderFloat("Dispersion", &dispersion, 0.0f, 0.20f);

        ImGui::ColorEdit3("Tint", glm::value_ptr(tint));

        ImGui::SliderFloat("Refract Strength", &refrStrength, 0.0f, 1.0f);
        ImGui::SliderFloat("Fresnel Strength", &fresnelStrength, 0.0f, 1.5f);

        ImGui::Separator();
        ImGui::SliderFloat("Cam Yaw (deg)", &camYawDeg, -180.0f, 180.0f);
        ImGui::SliderFloat("Cam Pitch (deg)", &camPitchDeg, -60.0f, 60.0f);
        ImGui::SliderFloat("Cam Distance", &camDist, 2.0f, 20.0f);

        ImGui::End();

        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        if (w <= 0 || h <= 0) continue;

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

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTex);

        glUseProgram(glassProg);
        glUniformMatrix4fv(g_uView,  1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(g_uProj,  1, GL_FALSE, glm::value_ptr(P));
        glUniform3fv(g_uCamPosW, 1, glm::value_ptr(camPos));
        glUniform1f(g_uIOR, ior);
        glUniform1f(g_uDisp, dispersion);
        glUniform3fv(g_uTint, 1, glm::value_ptr(tint));
        glUniform1f(g_uRefractStrength, refrStrength);
        glUniform1f(g_uFresnelStrength, fresnelStrength);

        float t = (float)glfwGetTime();

        // ---- Draw test.obj ----
        {
            glm::mat4 M = glm::mat4(1.0f);
            M = glm::translate(M, glm::vec3(0.0f, 0.0f, 0.0f));
            M = glm::rotate(M, t * 0.8f, glm::vec3(0, 1, 0));
            M = M * testMesh.baseModel;

            glUniformMatrix4fv(g_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glBindVertexArray(testMesh.vao);
            glDrawElements(GL_TRIANGLES, testMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // ---- Draw cap.obj ----
        {
            glm::mat4 M = glm::mat4(1.0f);
            M = glm::translate(M, glm::vec3(-2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, -t * 0.5f, glm::vec3(0, 1, 0));
            M = M * capMesh.baseModel;

            glUniformMatrix4fv(g_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glBindVertexArray(capMesh.vao);
            glDrawElements(GL_TRIANGLES, capMesh.indexCount, GL_UNSIGNED_INT, 0);
        }

        // ---- Draw Moon.obj ----
        {
            glm::mat4 M = glm::mat4(1.0f);
            M = glm::translate(M, glm::vec3(2.2f, 0.0f, 0.0f));
            M = glm::rotate(M, t * 0.25f, glm::vec3(0, 1, 0));
            M = M * moonMesh.baseModel;

            glUniformMatrix4fv(g_uModel, 1, GL_FALSE, glm::value_ptr(M));
            glBindVertexArray(moonMesh.vao);
            glDrawElements(GL_TRIANGLES, moonMesh.indexCount, GL_UNSIGNED_INT, 0);
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

    glDeleteProgram(glassProg);
    glDeleteProgram(skyProg);

    glDeleteTextures(1, &cubemapTex);

    destroyMesh(testMesh);
    destroyMesh(capMesh);
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
