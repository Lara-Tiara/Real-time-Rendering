#include "Mesh.h"

#include <cstddef>
#include <utility>

Mesh::Mesh(std::vector<Vertex> vertices,
           std::vector<unsigned int> indices,
           std::string meshName,
           std::string materialName)
    : m_vertices(std::move(vertices)),
      m_indices(std::move(indices)),
      m_meshName(std::move(meshName)),
      m_materialName(std::move(materialName)) {
    setupBuffers();
}

Mesh::~Mesh() {
    destroy();
}

Mesh::Mesh(Mesh&& other) noexcept
    : m_vertices(std::move(other.m_vertices)),
      m_indices(std::move(other.m_indices)),
      m_meshName(std::move(other.m_meshName)),
      m_materialName(std::move(other.m_materialName)),
      m_vao(other.m_vao),
      m_vbo(other.m_vbo),
      m_ebo(other.m_ebo) {
    other.m_vao = 0;
    other.m_vbo = 0;
    other.m_ebo = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        destroy();

        m_vertices = std::move(other.m_vertices);
        m_indices = std::move(other.m_indices);
        m_meshName = std::move(other.m_meshName);
        m_materialName = std::move(other.m_materialName);

        m_vao = other.m_vao;
        m_vbo = other.m_vbo;
        m_ebo = other.m_ebo;

        other.m_vao = 0;
        other.m_vbo = 0;
        other.m_ebo = 0;
    }
    return *this;
}

void Mesh::draw() const {
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_indices.size()), GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

bool Mesh::isValid() const {
    return m_vao != 0 && !m_indices.empty();
}

const std::string& Mesh::meshName() const {
    return m_meshName;
}

const std::string& Mesh::materialName() const {
    return m_materialName;
}

void Mesh::setupBuffers() {
    if (m_vertices.empty() || m_indices.empty()) {
        return;
    }

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(m_vertices.size() * sizeof(Vertex)),
        m_vertices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(m_indices.size() * sizeof(unsigned int)),
        m_indices.data(),
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, position))
    );

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, normal))
    );

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, texCoord))
    );

    glEnableVertexAttribArray(3);
    glVertexAttribPointer(
        3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
        reinterpret_cast<void*>(offsetof(Vertex, tangent))
    );

    glBindVertexArray(0);
}

void Mesh::destroy() {
    if (m_ebo != 0) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }
    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
}