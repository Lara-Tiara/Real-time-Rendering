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

static MaterialSet loadClothMaterial(const std::string& displayName, const std::string& folder, const std::string& prefix) {
    MaterialSet material;
    material.name = displayName;

    const std::string base = assetPath("textures/cloth/" + folder + "/");

    material.diffuse = std::make_unique<Texture>(base + prefix + "_diff_4k.png", TextureColorSpace::SRGB, true);
    material.roughness = std::make_unique<Texture>(base + prefix + "_rough_4k.png", TextureColorSpace::Linear, true);
    material.ao = std::make_unique<Texture>(base + prefix + "_ao_4k.png", TextureColorSpace::Linear, true);
    material.normalGL = std::make_unique<Texture>(base + prefix + "_nor_gl_4k.png", TextureColorSpace::Linear, true);
    material.anisoRotation = std::make_unique<Texture>(base + prefix + "_anisotropy_rotation_4k.png", TextureColorSpace::Linear, true);
    material.anisoStrength = std::make_unique<Texture>(base + prefix + "_anisotropy_strength_4k.png", TextureColorSpace::Linear, true);

    return material;
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

        std::vector<MaterialSet> materials;
        materials.push_back(loadClothMaterial("Crepe Satin", "crepe_satin", "crepe_satin"));
        materials.push_back(loadClothMaterial("Hessian 230", "hessian_230", "hessian_230"));
        materials.push_back(loadClothMaterial("Gingham Check", "gingham_check", "gingham_check"));

        std::vector<ModelEntry> models;

        // models.push_back({
        //     "Curtain Flannel",
        //     std::make_unique<Model>(assetPath("models/curtain_flannel/scene.gltf")),
        //     {}
        // });

        models.push_back({
            "Cloth On Cup",
            std::make_unique<Model>(assetPath("models/cloth_on_cup/scene.gltf")),
            {0}
        });

        models.push_back({
            "Cloth Ghost",
            std::make_unique<Model>(assetPath("models/cloth_ghost/scene.gltf")),
            {0}
        });

        models.push_back({
            "Crepe Satin Cloth",
            std::make_unique<Model>(assetPath("models/crepe_satin_4k/crepe_satin_4k.gltf")),
            {0}
        });

        for (const auto& entry : models) {
            entry.model->printMeshInfo();
        }

        int currentMaterialIndex = 0;
        int debugView = 0;

        float uvScale = 1.0f;
        float normalScale = 1.0f;
        float roughnessScale = 1.0f;
        float aoStrength = 1.0f;
        float anisotropyScale = 1.0f;

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

        while (!glfwWindowShouldClose(window)) {
            processInput(window);
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::Begin("Cloth Viewer");

            if (ImGui::BeginCombo("Material", materials[currentMaterialIndex].name.c_str())) {
                for (int i = 0; i < static_cast<int>(materials.size()); ++i) {
                    const bool selected = (i == currentMaterialIndex);
                    if (ImGui::Selectable(materials[i].name.c_str(), selected)) {
                        currentMaterialIndex = i;
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Combo("Debug View", &debugView, debugItems, IM_ARRAYSIZE(debugItems));

            ImGui::SliderFloat("UV Scale", &uvScale, 0.1f, 16.0f);
            ImGui::SliderFloat("Normal Scale", &normalScale, 0.0f, 3.0f);
            ImGui::SliderFloat("Roughness Scale", &roughnessScale, 0.1f, 3.0f);
            ImGui::SliderFloat("AO Strength", &aoStrength, 0.0f, 1.0f);
            ImGui::SliderFloat("Anisotropy Scale", &anisotropyScale, 0.0f, 3.0f);

            ImGui::Separator();

            ImGui::SliderFloat("Camera Distance", &cameraDistance, 2.0f, 12.0f);
            ImGui::SliderFloat("Model Scale", &modelScale, 0.1f, 3.0f);
            ImGui::SliderFloat("Model Yaw Offset", &modelYawOffset, -180.0f, 180.0f);
            ImGui::SliderFloat("Rotation Speed", &rotationSpeed, -180.0f, 180.0f);
            ImGui::SliderFloat("Model Spacing", &modelSpacing, 0.5f, 6.0f);

            ImGui::Separator();

            ImGui::SliderFloat("Light Elevation", &lightElevation, -85.0f, 85.0f);
            ImGui::SliderFloat("Light Azimuth", &lightAzimuth, -180.0f, 180.0f);

            ImGui::Text("Current Material: %s", materials[currentMaterialIndex].name.c_str());
            ImGui::Text("Models:");
            for (const auto& modelEntry : models) {
                ImGui::BulletText("%s", modelEntry.name.c_str());
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

            const MaterialSet& material = materials[currentMaterialIndex];

            glViewport(0, 0, fbWidth, fbHeight);
            glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            clothShader.use();

            clothShader.setMat4("uView", view);
            clothShader.setMat4("uProj", proj);

            clothShader.setVec3("uCameraPos", cameraPos);
            clothShader.setVec3("uLightDir", lightDir);
            clothShader.setVec3("uLightColor", lightColor);

            clothShader.setFloat("uUvScale", uvScale);
            clothShader.setFloat("uNormalScale", normalScale);
            clothShader.setFloat("uRoughnessScale", roughnessScale);
            clothShader.setFloat("uAoStrength", aoStrength);
            clothShader.setFloat("uAnisotropyScale", anisotropyScale);
            clothShader.setInt("uDebugView", debugView);

            clothShader.setInt("uDiffuseMap", 0);
            clothShader.setInt("uRoughnessMap", 1);
            clothShader.setInt("uAoMap", 2);
            clothShader.setInt("uNormalMap", 3);
            clothShader.setInt("uAnisoRotationMap", 4);
            clothShader.setInt("uAnisoStrengthMap", 5);

            material.diffuse->bind(0);
            material.roughness->bind(1);
            material.ao->bind(2);
            material.normalGL->bind(3);
            material.anisoRotation->bind(4);
            material.anisoStrength->bind(5);

            const float timeSeconds = static_cast<float>(glfwGetTime());
            const float sharedYaw = modelYawOffset + rotationSpeed * timeSeconds;

            const int modelCount = static_cast<int>(models.size());
            const float centerOffset = 0.5f * static_cast<float>(modelCount - 1);

            for (int i = 0; i < modelCount; ++i) {
                const Model& modelRef = *models[i].model;

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
                models[i].model->drawMeshIndices(models[i].meshIndices);
            }

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