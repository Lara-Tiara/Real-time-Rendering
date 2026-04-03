#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

struct Vertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 0.0f, 1.0f};
    glm::vec2 texCoord{0.0f};
    glm::vec3 tangent{0.0f};
};

class Mesh {
public:
    Mesh() = default;
    Mesh(std::vector<Vertex> vertices,
         std::vector<unsigned int> indices,
         std::string meshName = {},
         std::string materialName = {});
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void draw() const;
    bool isValid() const;

    const std::string& meshName() const;
    const std::string& materialName() const;

private:
    std::vector<Vertex> m_vertices;
    std::vector<unsigned int> m_indices;

    std::string m_meshName;
    std::string m_materialName;

    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_ebo = 0;

    void setupBuffers();
    void destroy();
};