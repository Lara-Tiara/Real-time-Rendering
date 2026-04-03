#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>

class Shader {
public:
    Shader() = default;
    Shader(const std::string& vertexPath, const std::string& fragmentPath);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);
    void use() const;

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] bool isValid() const;

    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, const glm::vec2& value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setMat3(const std::string& name, const glm::mat3& value) const;
    void setMat4(const std::string& name, const glm::mat4& value) const;

private:
    GLuint m_program = 0;

    static std::string readTextFile(const std::string& path);
    static GLuint compileStage(GLenum type, const std::string& source, const std::string& debugName);
    [[nodiscard]] GLint getLocation(const std::string& name) const;
    void destroy();
};