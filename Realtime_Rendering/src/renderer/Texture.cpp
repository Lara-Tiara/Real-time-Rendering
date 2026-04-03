#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <../stb/stb_image.h>

#include <stdexcept>
#include <string>

Texture::Texture(const std::string& path, TextureColorSpace colorSpace, bool flipVertically) {
    loadFromFile(path, colorSpace, flipVertically);
}

Texture::~Texture() {
    destroy();
}

bool Texture::loadFromFile(const std::string& path, TextureColorSpace colorSpace, bool flipVertically) {
    destroy();

    stbi_set_flip_vertically_on_load(flipVertically ? 1 : 0);

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);
    if (data == nullptr) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error(
            "Failed to load texture: " + path +
            (reason ? std::string("\n") + reason : std::string())
        );
    }

    const GLenum internalFormat = chooseInternalFormat(channels, colorSpace);
    const GLenum dataFormat = chooseDataFormat(channels);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        static_cast<GLint>(internalFormat),
        width,
        height,
        0,
        dataFormat,
        GL_UNSIGNED_BYTE,
        data
    );
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (dataFormat == GL_RED) {
        GLint swizzleMask[] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzleMask);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    stbi_image_free(data);

    m_texture = texture;
    m_width = width;
    m_height = height;
    m_channels = channels;
    m_path = path;
    m_colorSpace = colorSpace;

    return true;
}

void Texture::bind(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, m_texture);
}

void Texture::unbind(GLuint unit) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint Texture::id() const {
    return m_texture;
}

int Texture::width() const {
    return m_width;
}

int Texture::height() const {
    return m_height;
}

int Texture::channels() const {
    return m_channels;
}

const std::string& Texture::path() const {
    return m_path;
}

TextureColorSpace Texture::colorSpace() const {
    return m_colorSpace;
}

bool Texture::isValid() const {
    return m_texture != 0;
}

GLenum Texture::chooseInternalFormat(int channels, TextureColorSpace colorSpace) {
    switch (channels) {
        case 1:
            return GL_R8;
        case 2:
            return GL_RG8;
        case 3:
            return colorSpace == TextureColorSpace::SRGB ? GL_SRGB8 : GL_RGB8;
        case 4:
            return colorSpace == TextureColorSpace::SRGB ? GL_SRGB8_ALPHA8 : GL_RGBA8;
        default:
            throw std::runtime_error("Unsupported channel count: " + std::to_string(channels));
    }
}

GLenum Texture::chooseDataFormat(int channels) {
    switch (channels) {
        case 1:
            return GL_RED;
        case 2:
            return GL_RG;
        case 3:
            return GL_RGB;
        case 4:
            return GL_RGBA;
        default:
            throw std::runtime_error("Unsupported channel count: " + std::to_string(channels));
    }
}

void Texture::destroy() {
    if (m_texture != 0) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }

    m_width = 0;
    m_height = 0;
    m_channels = 0;
    m_path.clear();
}