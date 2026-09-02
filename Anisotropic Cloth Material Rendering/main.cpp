#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "src/renderer/Model.h"
#include "src/renderer/Shader.h"
#include "src/renderer/Texture.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct MaterialSet {
    std::string name;
    float localAnisotropyScale = 1.0f;
    std::unique_ptr<Texture> diffuse;
    std::unique_ptr<Texture> roughness;
    std::unique_ptr<Texture> ao;
    std::unique_ptr<Texture> normalGL;
    std::unique_ptr<Texture> anisoRotation;
    std::unique_ptr<Texture> anisoStrength;
};

struct ModelEntry {
    std::string name;
    std::unique_ptr<Model> model;
    std::vector<int> meshIndices;
    float uvScale = 1.0f;
    int materialIndex = 0;
    float aoStrength = 1.0f;
    float anisotropyScale = 1.0f;
};

struct ClothPreset {
    std::string name;
    std::string folder;
    std::string prefix;
    std::string modelRelativePath;
    std::vector<int> meshIndices;
    float defaultUvScale = 1.0f;
};

static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static std::string assetPath(const std::string& relative) {
    return std::string(ASSET_DIR) + "/" + relative;
}

static GLuint compileRawShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log(static_cast<size_t>(len > 0 ? len : 1), '\0');
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("Skybox shader compilation failed:\n" + log);
    }

    return shader;
}

static GLuint linkRawProgram(GLuint vs, GLuint fs) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::string log(static_cast<size_t>(len > 0 ? len : 1), '\0');
        glGetProgramInfoLog(program, len, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error("Skybox program link failed:\n" + log);
    }

    return program;
}


static MaterialSet loadMaterialSet(
    const std::string& displayName,
    float localAnisotropyScale,
    const std::string& diffusePath,
    const std::string& roughnessPath,
    const std::string& aoPath,
    const std::string& normalPath,
    const std::string& anisoRotationPath,
    const std::string& anisoStrengthPath
) {
    MaterialSet material;
    material.name = displayName;
    material.localAnisotropyScale = localAnisotropyScale;

    material.diffuse = std::make_unique<Texture>(assetPath(diffusePath), TextureColorSpace::SRGB, true);
    material.roughness = std::make_unique<Texture>(assetPath(roughnessPath), TextureColorSpace::Linear, true);
    material.ao = std::make_unique<Texture>(assetPath(aoPath), TextureColorSpace::Linear, true);
    material.normalGL = std::make_unique<Texture>(assetPath(normalPath), TextureColorSpace::Linear, true);
    material.anisoRotation = std::make_unique<Texture>(assetPath(anisoRotationPath), TextureColorSpace::Linear, true);
    material.anisoStrength = std::make_unique<Texture>(assetPath(anisoStrengthPath), TextureColorSpace::Linear, true);

    return material;
}

static std::vector<MaterialSet> loadMaterialVariants(const ClothPreset& cloth) {
    const std::string base = "textures/cloth/" + cloth.folder + "/";

    std::vector<MaterialSet> materials;
    materials.push_back(loadMaterialSet(
        "Reference",
        1.0f,
        base + cloth.prefix + "_diff_4k.png",
        base + cloth.prefix + "_rough_4k.png",
        base + cloth.prefix + "_ao_4k.png",
        base + cloth.prefix + "_nor_gl_4k.png",
        base + cloth.prefix + "_anisotropy_rotation_4k.png",
        base + cloth.prefix + "_anisotropy_strength_4k.png"
    ));

    materials.push_back(loadMaterialSet(
        "Generated",
        1.0f,
        base + cloth.prefix + "_diff_4k.png",
        base + cloth.prefix + "_rough_4k.png",
        base + cloth.prefix + "_ao_4k.png",
        base + cloth.prefix + "_nor_gl_4k.png",
        base + "generated/reconstructed_anisotropy_rotation.png",
        base + "generated/reconstructed_anisotropy_strength.png"
    ));

    materials.push_back(loadMaterialSet(
        "Baseline",
        0.0f,
        base + cloth.prefix + "_diff_4k.png",
        base + cloth.prefix + "_rough_4k.png",
        base + cloth.prefix + "_ao_4k.png",
        base + cloth.prefix + "_nor_gl_4k.png",
        base + cloth.prefix + "_anisotropy_rotation_4k.png",
        base + cloth.prefix + "_anisotropy_strength_4k.png"
    ));

    return materials;
}

static std::vector<ModelEntry> buildDisplayModels(const ClothPreset& cloth) {
    std::vector<ModelEntry> models;

    models.push_back({
        "Left Mesh",
        std::make_unique<Model>(assetPath(cloth.modelRelativePath)),
        cloth.meshIndices,
        cloth.defaultUvScale,
        0,
        1.0f,
        1.0f
    });

    models.push_back({
        "Center Mesh",
        std::make_unique<Model>(assetPath(cloth.modelRelativePath)),
        cloth.meshIndices,
        cloth.defaultUvScale,
        1,
        1.0f,
        1.0f
    });

    models.push_back({
        "Right Mesh",
        std::make_unique<Model>(assetPath(cloth.modelRelativePath)),
        cloth.meshIndices,
        cloth.defaultUvScale,
        2,
        1.0f,
        1.0f
    });

    return models;
}

static glm::vec3 directionFromAngles(float elevationDeg, float azimuthDeg) {
    const float elevation = glm::radians(elevationDeg);
    const float azimuth = glm::radians(azimuthDeg);

    const float x = std::cos(elevation) * std::cos(azimuth);
    const float y = std::sin(elevation);
    const float z = std::cos(elevation) * std::sin(azimuth);

    return glm::normalize(glm::vec3(x, y, z));
}

int main() {
    try {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        GLFWwindow* window = glfwCreateWindow(1600, 900, "Cloth Reference Viewer", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window.");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("Failed to initialize GLAD.");
        }

        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        Shader clothShader(
            assetPath("shaders/cloth_reference.vert"),
            assetPath("shaders/cloth_reference.frag")
        );

        const char* skyboxVS = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
out vec3 vTexDir;
uniform mat4 uView;
uniform mat4 uProj;
void main() {
    vTexDir = aPos;
    vec4 pos = uProj * uView * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}
)";

        const char* skyboxFS = R"(
#version 330 core
in vec3 vTexDir;
out vec4 FragColor;
uniform samplerCube uSkybox;
void main() {
    FragColor = texture(uSkybox, vTexDir);
}
)";

        GLuint skyboxProgram = 0;
        GLuint skyboxVAO = 0;
        GLuint skyboxVBO = 0;
        GLuint skyboxCubemap = 0;

        {
            GLuint vs = compileRawShader(GL_VERTEX_SHADER, skyboxVS);
            GLuint fs = compileRawShader(GL_FRAGMENT_SHADER, skyboxFS);
            skyboxProgram = linkRawProgram(vs, fs);
            glDeleteShader(vs);
            glDeleteShader(fs);


            const std::vector<std::string> skyboxFaces = {
                assetPath("textures/skybox/right.jpg"),
                assetPath("textures/skybox/left.jpg"),
                assetPath("textures/skybox/top.jpg"),
                assetPath("textures/skybox/bottom.jpg"),
                assetPath("textures/skybox/front.jpg"),
                assetPath("textures/skybox/back.jpg")
            };


            glUseProgram(skyboxProgram);
            glUniform1i(glGetUniformLocation(skyboxProgram, "uSkybox"), 0);
        }

        std::vector<ClothPreset> clothPresets;
        clothPresets.push_back({
            "Crepe Satin",
            "crepe_satin",
            "crepe_satin",
            "models/crepe_satin_4k/crepe_satin_4k.gltf",
            {0},
            1.0f
        });
        clothPresets.push_back({
            "Gingham Check",
            "gingham_check",
            "gingham_check",
            "models/gingham_check_4k/gingham_check_4k.gltf",
            {0},
            1.0f
        });
        clothPresets.push_back({
            "Hessian 230",
            "hessian_230",
            "hessian_230",
            "models/hessian_230_4k/hessian_230_4k.gltf",
            {0},
            1.0f
        });

        int currentClothIndex = 0;
        std::vector<MaterialSet> materials = loadMaterialVariants(clothPresets[currentClothIndex]);
        std::vector<ModelEntry> models = buildDisplayModels(clothPresets[currentClothIndex]);

        for (const auto& entry : models) {
            entry.model->printMeshInfo();
        }

        int debugView = 0;

        float globalUvScale = 1.0f;
        float normalScale = 1.0f;
        float roughnessScale = 1.0f;

        float cameraDistance = 5.2f;
        float modelScale = 1.0f;
        float modelYawOffset = 0.0f;
        float rotationSpeed = 20.0f;
        float modelSpacing = 2.4f;

        float lightElevation = 30.0f;
        float lightAzimuth = -45.0f;

        const char* debugItems[] = {
            "Final Shaded",
            "Diffuse",
            "Roughness",
            "AO",
            "Normal",
            "Rotation",
            "Strength"
        };

        const char* materialItems[] = {
            "Reference",
            "Generated",
            "Baseline"
        };

        const char* clothItems[] = {
            "Crepe Satin",
            "Gingham Check",
            "Hessian 230"
        };

        while (!glfwWindowShouldClose(window)) {
            processInput(window);
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::Begin("Cloth Viewer");

            if (ImGui::Combo("Cloth", &currentClothIndex, clothItems, IM_ARRAYSIZE(clothItems))) {
                materials = loadMaterialVariants(clothPresets[currentClothIndex]);
                models = buildDisplayModels(clothPresets[currentClothIndex]);
                for (const auto& entry : models) {
                    entry.model->printMeshInfo();
                }
            }

            ImGui::Text("Three identical meshes with switchable Reference / Generated / Baseline variants.");
            ImGui::Text("Current cloth folder: %s", clothPresets[currentClothIndex].folder.c_str());
            ImGui::Combo("Debug View", &debugView, debugItems, IM_ARRAYSIZE(debugItems));

            ImGui::Separator();
            ImGui::Text("Material Variant Per Mesh");
            for (int i = 0; i < static_cast<int>(models.size()); ++i) {
                std::string comboLabel = models[i].name + " Material";
                ImGui::Combo(comboLabel.c_str(), &models[i].materialIndex, materialItems, IM_ARRAYSIZE(materialItems));
            }

            ImGui::Separator();
            ImGui::SliderFloat("Global UV Scale", &globalUvScale, 0.001f, 16.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::Text("Per-Mesh UV Scale");
            for (int i = 0; i < static_cast<int>(models.size()); ++i) {
                std::string label = models[i].name + " UV";
                ImGui::SliderFloat(label.c_str(), &models[i].uvScale, 0.001f, 4.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            }

            ImGui::Separator();
            ImGui::SliderFloat("Normal Scale", &normalScale, 0.0f, 3.0f);
            ImGui::SliderFloat("Roughness Scale", &roughnessScale, 0.1f, 3.0f);

            ImGui::Separator();
            ImGui::Text("Per-Mesh AO Strength");
            for (int i = 0; i < static_cast<int>(models.size()); ++i) {
                std::string label = models[i].name + " AO";
                ImGui::SliderFloat(label.c_str(), &models[i].aoStrength, 0.0f, 1.0f);
            }

            ImGui::Text("Per-Mesh Anisotropy Scale");
            for (int i = 0; i < static_cast<int>(models.size()); ++i) {
                std::string label = models[i].name + " Anisotropy";
                ImGui::SliderFloat(label.c_str(), &models[i].anisotropyScale, 0.0f, 10.0f);
            }

            ImGui::Separator();
            ImGui::SliderFloat("Camera Distance", &cameraDistance, 2.0f, 12.0f);
            ImGui::SliderFloat("Model Scale", &modelScale, 0.1f, 3.0f);
            ImGui::SliderFloat("Model Yaw Offset", &modelYawOffset, -180.0f, 180.0f);
            ImGui::SliderFloat("Rotation Speed", &rotationSpeed, -180.0f, 180.0f);
            ImGui::SliderFloat("Model Spacing", &modelSpacing, 0.5f, 6.0f);

            ImGui::Separator();
            ImGui::SliderFloat("Light Elevation", &lightElevation, -85.0f, 85.0f);
            ImGui::SliderFloat("Light Azimuth", &lightAzimuth, -180.0f, 180.0f);

            ImGui::Separator();
            ImGui::Text("Current assignments:");
            for (const auto& modelEntry : models) {
                ImGui::BulletText(
                    "%s -> %s | AO %.2f | Aniso %.2f",
                    modelEntry.name.c_str(),
                    materials[modelEntry.materialIndex].name.c_str(),
                    modelEntry.aoStrength,
                    modelEntry.anisotropyScale
                );
            }

            ImGui::End();

            int fbWidth = 0;
            int fbHeight = 0;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            const float aspect = (fbHeight > 0) ? static_cast<float>(fbWidth) / static_cast<float>(fbHeight) : 1.0f;

            glm::vec3 cameraPos(0.0f, 0.0f, cameraDistance);
            glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);

            glm::vec3 lightDir = directionFromAngles(lightElevation, lightAzimuth);
            glm::vec3 lightColor(3.0f, 3.0f, 3.0f);

            glViewport(0, 0, fbWidth, fbHeight);
            glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            clothShader.use();
            clothShader.setMat4("uView", view);
            clothShader.setMat4("uProj", proj);
            clothShader.setVec3("uCameraPos", cameraPos);
            clothShader.setVec3("uLightDir", lightDir);
            clothShader.setVec3("uLightColor", lightColor);
            clothShader.setFloat("uNormalScale", normalScale);
            clothShader.setFloat("uRoughnessScale", roughnessScale);
            clothShader.setInt("uDebugView", debugView);

            clothShader.setInt("uDiffuseMap", 0);
            clothShader.setInt("uRoughnessMap", 1);
            clothShader.setInt("uAoMap", 2);
            clothShader.setInt("uNormalMap", 3);
            clothShader.setInt("uAnisoRotationMap", 4);
            clothShader.setInt("uAnisoStrengthMap", 5);

            const float timeSeconds = static_cast<float>(glfwGetTime());
            const float sharedYaw = modelYawOffset + rotationSpeed * timeSeconds;

            const int modelCount = static_cast<int>(models.size());
            const float centerOffset = 0.5f * static_cast<float>(modelCount - 1);

            for (int i = 0; i < modelCount; ++i) {
                const ModelEntry& entry = models[i];
                const MaterialSet& material = materials[entry.materialIndex];
                const Model& modelRef = *entry.model;

                material.diffuse->bind(0);
                material.roughness->bind(1);
                material.ao->bind(2);
                material.normalGL->bind(3);
                material.anisoRotation->bind(4);
                material.anisoStrength->bind(5);

                clothShader.setFloat("uAoStrength", entry.aoStrength);
                clothShader.setFloat("uAnisotropyScale", entry.anisotropyScale * material.localAnisotropyScale);

                const glm::vec3 center = modelRef.center();
                const float radius = modelRef.radius();
                const float autoScale = radius > 1e-5f ? (1.0f / radius) : 1.0f;
                const float finalScale = autoScale * modelScale;

                const float xOffset = (static_cast<float>(i) - centerOffset) * modelSpacing;

                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(xOffset, 0.0f, 0.0f));
                model = glm::rotate(model, glm::radians(sharedYaw), glm::vec3(0.0f, 1.0f, 0.0f));
                model = glm::scale(model, glm::vec3(finalScale));
                model = glm::translate(model, -center);

                clothShader.setMat4("uModel", model);
                clothShader.setFloat("uUvScale", globalUvScale * entry.uvScale);

                entry.model->drawMeshIndices(entry.meshIndices);
            }

            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);

            glUseProgram(skyboxProgram);
            glm::mat4 skyboxView = glm::mat4(glm::mat3(view));
            glUniformMatrix4fv(glGetUniformLocation(skyboxProgram, "uView"), 1, GL_FALSE, glm::value_ptr(skyboxView));
            glUniformMatrix4fv(glGetUniformLocation(skyboxProgram, "uProj"), 1, GL_FALSE, glm::value_ptr(proj));

            glBindVertexArray(skyboxVAO);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, skyboxCubemap);
            glDrawArrays(GL_TRIANGLES, 0, 36);
            glBindVertexArray(0);

            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(window);
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
