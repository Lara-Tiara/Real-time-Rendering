#include "Model.h"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    std::string toLowerCopy(const std::string& s) {
        std::string out = s;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }

    bool containsAnyKeyword(const std::string& haystack,
                            const std::vector<std::string>& keywords) {
        if (keywords.empty()) {
            return true;
        }

        const std::string lowerHaystack = toLowerCopy(haystack);
        for (const std::string& keyword : keywords) {
            if (keyword.empty()) {
                continue;
            }
            if (lowerHaystack.find(toLowerCopy(keyword)) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
}

Model::Model(const std::string& path) {
    loadFromFile(path);
}

void Model::loadFromFile(const std::string& path) {
    m_path = path;
    m_meshes.clear();
    m_boundsMin = glm::vec3(0.0f);
    m_boundsMax = glm::vec3(0.0f);
    m_hasBounds = false;

    Assimp::Importer importer;

    const unsigned int flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_SortByPType |
        aiProcess_PreTransformVertices |
        aiProcess_FlipUVs;

    const aiScene* scene = importer.ReadFile(path, flags);

    if (scene == nullptr || scene->mRootNode == nullptr || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0) {
        throw std::runtime_error("Failed to load model: " + path + "\n" + importer.GetErrorString());
    }

    processNode(scene->mRootNode, scene);

    if (m_meshes.empty()) {
        throw std::runtime_error("Model contains no drawable meshes: " + path);
    }
}

void Model::draw() const {
    for (const Mesh& mesh : m_meshes) {
        mesh.draw();
    }
}

void Model::drawMeshIndices(const std::vector<int>& meshIndices) const {
    if (meshIndices.empty()) {
        draw();
        return;
    }

    for (int index : meshIndices) {
        if (index >= 0 && index < static_cast<int>(m_meshes.size())) {
            m_meshes[index].draw();
        }
    }
}

bool Model::isValid() const {
    return !m_meshes.empty();
}

const std::string& Model::path() const {
    return m_path;
}

glm::vec3 Model::center() const {
    if (!m_hasBounds) {
        return glm::vec3(0.0f);
    }
    return 0.5f * (m_boundsMin + m_boundsMax);
}

float Model::radius() const {
    if (!m_hasBounds) {
        return 1.0f;
    }
    return glm::length(0.5f * (m_boundsMax - m_boundsMin));
}

void Model::printMeshInfo() const {
    std::cout << "\n=== Model Mesh Info: " << m_path << " ===\n";
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        std::cout << "[" << i << "] mesh=\"" << m_meshes[i].meshName()
                  << "\" material=\"" << m_meshes[i].materialName() << "\"\n";
    }
    std::cout << "===============================\n";
}

void Model::processNode(aiNode* node, const aiScene* scene) {
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        m_meshes.push_back(processMesh(mesh, scene));
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        processNode(node->mChildren[i], scene);
    }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene) {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    vertices.reserve(mesh->mNumVertices);

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        Vertex v;

        v.position = glm::vec3(
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
        );

        if (mesh->HasNormals()) {
            v.normal = glm::vec3(
                mesh->mNormals[i].x,
                mesh->mNormals[i].y,
                mesh->mNormals[i].z
            );
        }

        if (mesh->mTextureCoords[0] != nullptr) {
            v.texCoord = glm::vec2(
                mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y
            );
        }

        if (mesh->HasTangentsAndBitangents()) {
            v.tangent = glm::vec3(
                mesh->mTangents[i].x,
                mesh->mTangents[i].y,
                mesh->mTangents[i].z
            );
        }

        updateBounds(v.position);
        vertices.push_back(v);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; ++j) {
            indices.push_back(face.mIndices[j]);
        }
    }

    std::string meshName = mesh->mName.C_Str();
    std::string materialName;

    if (mesh->mMaterialIndex < scene->mNumMaterials) {
        aiString matName;
        if (scene->mMaterials[mesh->mMaterialIndex]->Get(AI_MATKEY_NAME, matName) == AI_SUCCESS) {
            materialName = matName.C_Str();
        }
    }

    return Mesh(std::move(vertices), std::move(indices), meshName, materialName);
}

void Model::updateBounds(const glm::vec3& p) {
    if (!m_hasBounds) {
        m_boundsMin = p;
        m_boundsMax = p;
        m_hasBounds = true;
        return;
    }

    m_boundsMin.x = std::min(m_boundsMin.x, p.x);
    m_boundsMin.y = std::min(m_boundsMin.y, p.y);
    m_boundsMin.z = std::min(m_boundsMin.z, p.z);

    m_boundsMax.x = std::max(m_boundsMax.x, p.x);
    m_boundsMax.y = std::max(m_boundsMax.y, p.y);
    m_boundsMax.z = std::max(m_boundsMax.z, p.z);
}