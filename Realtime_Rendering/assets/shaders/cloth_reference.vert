#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aTangent;

out VS_OUT {
    vec3 worldPos;
    vec3 worldNormal;
    vec2 uv;
    mat3 TBN;
} vs_out;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform float uUvScale;

vec3 buildFallbackTangent(vec3 n) {
    vec3 up = abs(n.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(up, n));
}

void main() {
    vec4 worldPos4 = uModel * vec4(aPos, 1.0);
    vs_out.worldPos = worldPos4.xyz;

    mat3 normalMatrix = transpose(inverse(mat3(uModel)));
    vec3 N = normalize(normalMatrix * aNormal);

    vec3 T = aTangent;
    if (length(T) < 1e-5) {
        T = buildFallbackTangent(N);
    } else {
        T = normalize(mat3(uModel) * T);
        T = normalize(T - dot(T, N) * N);
        if (length(T) < 1e-5) {
            T = buildFallbackTangent(N);
        }
    }

    vec3 B = normalize(cross(N, T));

    vs_out.worldNormal = N;
    vs_out.uv = aTexCoord * uUvScale;
    vs_out.TBN = mat3(T, B, N);

    gl_Position = uProj * uView * worldPos4;
}