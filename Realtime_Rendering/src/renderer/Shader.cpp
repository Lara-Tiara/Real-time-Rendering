#include "Shader.h"

#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath) {
    loadFromFiles(vertexPath, fragmentPath);
}

Shader::~Shader() {
    destroy();
}

void Shader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) {
    const std::string vertexSource = readTextFile(vertexPath);
    const std::string fragmentSource = readTextFile(fragmentPath);

    const GLuint vertexShader = compileStage(GL_VERTEX_SHADER, vertexSource, vertexPath);
    const GLuint fragmentShader = compileStage(GL_FRAGMENT_SHADER, fragmentSource, fragmentPath);

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE) {
        GLint logLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        std::string log(static_cast<size_t>(logLength), '\0');
        if (logLength > 0) {
            glGetProgramInfoLog(program, logLength, nullptr, log.data());
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        glDeleteProgram(program);

        throw std::runtime_error("Failed to link shader program:\n" + log);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    destroy();
    m_program = program;
}

void Shader::use() const {
    glUseProgram(m_program);
}

GLuint Shader::id() const {
    return m_program;
}

bool Shader::isValid() const {
    return m_program != 0;
}

void Shader::setBool(const std::string& name, bool value) const {
    glUniform1i(getLocation(name), static_cast<int>(value));
}

void Shader::setInt(const std::string& name, int value) const {
    glUniform1i(getLocation(name), value);
}

void Shader::setFloat(const std::string& name, float value) const {
    glUniform1f(getLocation(name), value);
}

void Shader::setVec2(const std::string& name, const glm::vec2& value) const {
    glUniform2fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setVec3(const std::string& name, const glm::vec3& value) const {
    glUniform3fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setVec4(const std::string& name, const glm::vec4& value) const {
    glUniform4fv(getLocation(name), 1, glm::value_ptr(value));
}

void Shader::setMat3(const std::string& name, const glm::mat3& value) const {
    glUniformMatrix3fv(getLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setMat4(const std::string& name, const glm::mat4& value) const {
    glUniformMatrix4fv(getLocation(name), 1, GL_FALSE, glm::value_ptr(value));
}

std::string Shader::readTextFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint Shader::compileStage(GLenum type, const std::string& source, const std::string& debugName) {
    const GLuint shader = glCreateShader(type);
    const char* sourcePtr = source.c_str();
    glShaderSource(shader, 1, &sourcePtr, nullptr);
    glCompileShader(shader);

    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE) {
        GLint logLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        std::string log(static_cast<size_t>(logLength), '\0');
        if (logLength > 0) {
            glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        }

        glDeleteShader(shader);

        const std::string shaderType =
            (type == GL_VERTEX_SHADER) ? "vertex" :
            (type == GL_FRAGMENT_SHADER) ? "fragment" :
            "unknown";

        throw std::runtime_error("Failed to compile " + shaderType + " shader: " + debugName + "\n" + log);
    }

    return shader;
}

GLint Shader::getLocation(const std::string& name) const {
    return glGetUniformLocation(m_program, name.c_str());
}

void Shader::destroy() {
    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}