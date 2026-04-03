#pragma once

#include "Mesh.h"

#include <glm/glm.hpp>

#include <string>
#include <vector>

struct aiNode;
struct aiMesh;
struct aiScene;

class Model {
public:
    Model() = default;
    explicit Model(const std::string& path);

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void loadFromFile(const std::string& path);
    void draw() const;
    void drawMeshIndices(const std::vector<int>& meshIndices) const;

    bool isValid() const;
    const std::string& path() const;

    glm::vec3 center() const;
    float radius() const;

    void printMeshInfo() const;

private:
    std::string m_path;
    std::vector<Mesh> m_meshes;

    glm::vec3 m_boundsMin{0.0f};
    glm::vec3 m_boundsMax{0.0f};
    bool m_hasBounds = false;

    void processNode(aiNode* node, const aiScene* scene);
    Mesh processMesh(aiMesh* mesh, const aiScene* scene);
    void updateBounds(const glm::vec3& p);
};