// 0 = Blinn-Phong, 1 = Toon, 2 = Oren-Nayar, 3 = Cook-Torrance

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cfloat>
#include <cmath>
#include <algorithm>

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

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1400, 720, "4 Teapots: Blinn-Phong / Toon / Oren-Nayar / Cook-Torrance", nullptr, nullptr);
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

    // ---- ImGui init ----
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // ---- Load model ----
    //std::string modelPath = std::string(ASSET_DIR) + "/teapot.gltf";
    std::string modelPath = std::string(ASSET_DIR) + "/test.obj";
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

// 0=BlinnPhong, 1=Toon, 2=OrenNayar, 3=CookTorrance
uniform int  uModelType;

uniform vec3  uAlbedo;
uniform vec3  uLightPosW;
uniform vec3  uLightColor;
uniform vec3  uCamPosW;
uniform float uLightIntensity;

uniform float uKs;
uniform float uShininess;
uniform float uRoughness;
uniform float uToonLevels;

uniform float uMetallic;
uniform float uAO;
uniform float uRimStrength;
uniform float uSpecThreshold;

const float PI = 3.14159265359;

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
    float x = ndotl * levels;
    float band = floor(x) / levels;

    float edge = fract(x);
    float smoothW = 0.10;
    float bandNext = min(1.0, (floor(x) + 1.0) / levels);
    float t = smoothstep(0.5 - smoothW, 0.5 + smoothW, edge);
    float q = mix(band, bandNext, t);

    vec3 H = normalize(L + V);
    float s = pow(max(dot(N, H), 0.0), uShininess);

    float specThreshold = uSpecThreshold;
    float specSoft = 0.05;
    float specBand = smoothstep(specThreshold - specSoft, specThreshold + specSoft, s);

    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    float rimStrength = uRimStrength;

    vec3 diffuse  = uAlbedo * q;
    vec3 specular = uKs * specBand * vec3(1.0);
    vec3 rimCol   = rimStrength * rim * uAlbedo;

    return (diffuse + specular + rimCol) * uLightColor;
}

// Oren–Nayar diffuse only
vec3 orenNayar(vec3 N, vec3 V, vec3 L) {
    float ndotl = max(dot(N, L), 0.0);
    float ndotv = max(dot(N, V), 0.0);
    if (ndotl <= 0.0 || ndotv <= 0.0) return vec3(0.0);

    float sigma = uRoughness * 1.57079632679;
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
    return (uAlbedo * oren) * uLightColor;
}

// -------------------- Cook–Torrance (GGX) --------------------
float D_GGX(float NdotH, float a) {
    float a2 = a * a;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float G_SmithGGX(float NdotV, float NdotL, float a) {
    // Schlick-GGX geometry term
    float k = (a + 1.0);
    k = (k * k) / 8.0;

    float gV = NdotV / (NdotV * (1.0 - k) + k);
    float gL = NdotL / (NdotL * (1.0 - k) + k);
    return gV * gL;
}

vec3 F_Schlick(vec3 F0, float VdotH) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

vec3 cookTorrance(vec3 N, vec3 V, vec3 L) {
    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

    vec3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    float a = max(0.03, uRoughness * uRoughness);

    float D = D_GGX(NdotH, a);
    float G = G_SmithGGX(NdotV, NdotL, a);

    vec3 F0 = mix(vec3(0.04), uAlbedo, uMetallic);
    vec3 F  = F_Schlick(F0, VdotH);

    vec3 spec = (D * G * F) / max(0.001, 4.0 * NdotV * NdotL);

    vec3 kd   = (vec3(1.0) - F) * (1.0 - uMetallic);
    vec3 diff = kd * uAlbedo / PI;

    vec3 brdf = diff + (uKs * spec);

    return (brdf * NdotL) * uLightColor;
}
// ------------------------------------------------------------

void main() {
    vec3 N = normalize(vNrmW);
    vec3 V = normalize(uCamPosW - vPosW);
    vec3 L = normalize(uLightPosW - vPosW);

    vec3 c;
    if (uModelType == 0)      c = blinnPhong(N, V, L);
    else if (uModelType == 1) c = toon(N, V, L);
    else if (uModelType == 2) c = orenNayar(N, V, L);
    else                      c = cookTorrance(N, V, L);

    float ao = (uModelType == 3) ? uAO : 1.0;
    vec3 ambient = 0.08 * uAlbedo * uLightColor * ao;
    vec3 hdr = ambient + c;

    // exposure / artistic boost
    hdr *= 1.2;

    // Reinhard tone mapping
    vec3 mapped = hdr / (hdr + vec3(1.0));

    // gamma encode
    vec3 finalCol = pow(max(mapped, vec3(0.0)), vec3(1.0/2.2));

    FragColor = vec4(finalCol, 1.0);
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
    const GLint loc_uLightIntensity = glGetUniformLocation(prog, "uLightIntensity");

    const GLint loc_uMetallic     = glGetUniformLocation(prog, "uMetallic");
    const GLint loc_uAO           = glGetUniformLocation(prog, "uAO");
    const GLint loc_uRimStrength  = glGetUniformLocation(prog, "uRimStrength");
    const GLint loc_uSpecThreshold= glGetUniformLocation(prog, "uSpecThreshold");

    const glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
    const glm::vec3 albedo(0.75f, 0.78f, 0.85f);

    const float offsets[4] = {-2.4f, -0.8f, 0.8f, 2.4f};

    struct MaterialUI {
        glm::vec3 albedo;
        float ks;
        float shininess;
        float roughness;
        float toonLevels;
        float metallic;
        float ao;
        float rimStrength;
        float specThreshold;
    };

    MaterialUI mats[4] = {
        // 0 Blinn-Phong
        { glm::vec3(0.75f, 0.78f, 0.85f), 0.55f, 96.0f, 0.35f, 4.0f, 0.0f, 1.0f, 0.20f, 0.50f },
        // 1 Toon
        { glm::vec3(0.75f, 0.78f, 0.85f), 0.25f, 48.0f, 0.35f, 4.0f, 0.0f, 1.0f, 0.20f, 0.50f },
        // 2 Oren-Nayar
        { glm::vec3(0.75f, 0.78f, 0.85f), 0.00f,  1.0f, 0.75f, 4.0f, 0.0f, 1.0f, 0.0f,  0.0f  },
        // 3 Cook-Torrance
        { glm::vec3(0.75f, 0.78f, 0.85f), 1.00f,  1.0f, 0.35f, 4.0f, 0.0f, 1.0f, 0.0f,  0.0f  }
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Reflectance Controls");

        if (ImGui::CollapsingHeader("0) Blinn-Phong", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo##BP", glm::value_ptr(mats[0].albedo));
            ImGui::SliderFloat("Ks##BP", &mats[0].ks, 0.0f, 2.0f);
            ImGui::SliderFloat("Shininess##BP", &mats[0].shininess, 1.0f, 256.0f);
        }

        if (ImGui::CollapsingHeader("1) Toon", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo##Toon", glm::value_ptr(mats[1].albedo));
            ImGui::SliderFloat("Toon Levels##Toon", &mats[1].toonLevels, 2.0f, 8.0f);
            ImGui::SliderFloat("Ks##Toon", &mats[1].ks, 0.0f, 2.0f);
            ImGui::SliderFloat("Shininess##Toon", &mats[1].shininess, 1.0f, 256.0f);
            ImGui::SliderFloat("Spec Threshold##Toon", &mats[1].specThreshold, 0.0f, 1.0f);
            ImGui::SliderFloat("Rim Strength##Toon", &mats[1].rimStrength, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("2) Oren-Nayar", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo##ON", glm::value_ptr(mats[2].albedo));
            ImGui::SliderFloat("Roughness##ON", &mats[2].roughness, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("3) Cook-Torrance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo##CT", glm::value_ptr(mats[3].albedo));
            ImGui::SliderFloat("Metallic##CT", &mats[3].metallic, 0.0f, 1.0f);
            ImGui::SliderFloat("Roughness##CT", &mats[3].roughness, 0.02f, 1.0f);
            ImGui::SliderFloat("AO##CT", &mats[3].ao, 0.0f, 1.0f);
            ImGui::SliderFloat("Spec Scale (Ks)##CT", &mats[3].ks, 0.0f, 2.0f);
        }

        ImGui::End();


        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, 1);

        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        if (w <= 0 || h <= 0) continue;

        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::vec3 camPos(0.0f, 1.1f, 4.2f);
        glm::mat4 P = glm::perspective(glm::radians(60.0f), (float)w / (float)h, 0.1f, 100.0f);
        glm::mat4 V = glm::lookAt(camPos, glm::vec3(0.0f, 0.3f, 0.0f), glm::vec3(0, 1, 0));

        glm::vec3 lightPos(100.0f, 25.0f, 16.0f);

        glUseProgram(prog);
        glUniformMatrix4fv(loc_uView, 1, GL_FALSE, glm::value_ptr(V));
        glUniformMatrix4fv(loc_uProj, 1, GL_FALSE, glm::value_ptr(P));

        glUniform3fv(loc_uLightPosW, 1, glm::value_ptr(lightPos));
        glUniform3fv(loc_uLightColor, 1, glm::value_ptr(lightColor));
        glUniform3fv(loc_uCamPosW, 1, glm::value_ptr(camPos));

        glUniform1f(loc_uLightIntensity, 8.0f);

        glBindVertexArray(vao);

        float t = (float)glfwGetTime();
        const float omega = 0.8f;

        for (int i = 0; i < 4; ++i) {
            int modelType = i; // 0=Blinn,1=Toon,2=Oren,3=Cook
            glUniform1i(loc_uModelType, modelType);

            glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(offsets[i], 0.0f, 0.0f));
            glm::mat4 R = glm::rotate(glm::mat4(1.0f), t * omega, glm::vec3(0, 1, 0));
            glm::mat4 M = T * R * baseModel;

            glUniformMatrix4fv(loc_uModel, 1, GL_FALSE, glm::value_ptr(M));

            MaterialUI& m = mats[i];

            glUniform3fv(loc_uAlbedo, 1, glm::value_ptr(m.albedo));
            glUniform1f(loc_uKs, m.ks);
            glUniform1f(loc_uShininess, m.shininess);
            glUniform1f(loc_uRoughness, m.roughness);
            glUniform1f(loc_uToonLevels, m.toonLevels);

            glUniform1f(loc_uMetallic, m.metallic);
            glUniform1f(loc_uAO, m.ao);
            glUniform1f(loc_uRimStrength, m.rimStrength);
            glUniform1f(loc_uSpecThreshold, m.specThreshold);

            glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // ---- Cleanup ----
    glDeleteProgram(prog);
    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
