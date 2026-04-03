#pragma once

#include <glad/glad.h>

#include <string>

enum class TextureColorSpace {
    Linear,
    SRGB
};

class Texture {
public:
    Texture() = default;
    Texture(const std::string& path, TextureColorSpace colorSpace, bool flipVertically = true);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    bool loadFromFile(const std::string& path, TextureColorSpace colorSpace, bool flipVertically = true);

    void bind(GLuint unit) const;
    static void unbind(GLuint unit);

    [[nodiscard]] GLuint id() const;
    [[nodiscard]] int width() const;
    [[nodiscard]] int height() const;
    [[nodiscard]] int channels() const;
    [[nodiscard]] const std::string& path() const;
    [[nodiscard]] TextureColorSpace colorSpace() const;
    [[nodiscard]] bool isValid() const;

private:
    GLuint m_texture = 0;
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::string m_path;
    TextureColorSpace m_colorSpace = TextureColorSpace::Linear;

    static GLenum chooseInternalFormat(int channels, TextureColorSpace colorSpace);
    static GLenum chooseDataFormat(int channels);
    void destroy();
};