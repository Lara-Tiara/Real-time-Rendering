#version 330 core

out vec4 FragColor;

in VS_OUT {
    vec3 worldPos;
    vec3 worldNormal;
    vec2 uv;
    mat3 TBN;
} fs_in;

uniform sampler2D uDiffuseMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uAoMap;
uniform sampler2D uNormalMap;
uniform sampler2D uAnisoRotationMap;
uniform sampler2D uAnisoStrengthMap;

uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

uniform float uNormalScale;
uniform float uRoughnessScale;
uniform float uAoStrength;
uniform float uAnisotropyScale;
uniform int uDebugView;

const float PI = 3.14159265359;

vec3 decodeNormal(vec3 packedNormal) {
    vec3 n = packedNormal * 2.0 - 1.0;
    n.xy *= uNormalScale;
    return normalize(n);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float sampleScalarMap(sampler2D tex, vec2 uv) {
    return texture(tex, uv).r;
}

vec3 visualizeNormal(vec3 n) {
    return n * 0.5 + 0.5;
}

void main() {
    vec2 uv = fs_in.uv;

    vec3 albedo = texture(uDiffuseMap, uv).rgb;

    float roughness = sampleScalarMap(uRoughnessMap, uv);
    roughness = clamp(roughness * uRoughnessScale, 0.02, 1.0);

    float ao = sampleScalarMap(uAoMap, uv);
    ao = mix(1.0, ao, clamp(uAoStrength, 0.0, 1.0));

    float rotationTex = sampleScalarMap(uAnisoRotationMap, uv);
    float strengthTex = sampleScalarMap(uAnisoStrengthMap, uv);
    float anisoStrength = clamp(strengthTex * uAnisotropyScale, 0.0, 1.0);

    vec3 tangentNormal = decodeNormal(texture(uNormalMap, uv).rgb);
    vec3 N = normalize(fs_in.TBN * tangentNormal);
    vec3 V = normalize(uCameraPos - fs_in.worldPos);
    vec3 L = normalize(-uLightDir);
    vec3 H = normalize(V + L);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = max(dot(N, H), 1e-4);

    vec3 T = normalize(fs_in.TBN[0]);
    vec3 B = normalize(fs_in.TBN[1]);

    float theta = rotationTex * 2.0 * PI;
    vec3 Trot = normalize(T * cos(theta) + B * sin(theta));
    vec3 Brot = normalize(cross(N, Trot));

    float TdotH = dot(Trot, H);
    float BdotH = dot(Brot, H);

    float alphaBase = max(0.02, roughness * roughness);
    float alphaT = max(0.02, mix(alphaBase, alphaBase * 0.2, anisoStrength));
    float alphaB = max(0.02, mix(alphaBase, alphaBase * 2.5, anisoStrength));

    float denom = max(NdotH * NdotH, 1e-5);
    float exponent = -((TdotH * TdotH) / (alphaT * alphaT) + (BdotH * BdotH) / (alphaB * alphaB)) / denom;
    float D = exp(exponent) / max(PI * alphaT * alphaB * denom * denom, 1e-5);

    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float Gv = NdotV / max(NdotV * (1.0 - k) + k, 1e-5);
    float Gl = NdotL / max(NdotL * (1.0 - k) + k, 1e-5);
    float G = Gv * Gl;

    vec3 F0 = vec3(0.04);
    vec3 F = fresnelSchlick(saturate(dot(H, V)), F0);

    vec3 diffuse = albedo / PI;
    vec3 specular = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-5);

    vec3 color = (diffuse + specular) * uLightColor * NdotL;
    color *= ao;

    if (uDebugView == 1) {
        FragColor = vec4(albedo, 1.0);
        return;
    }
    if (uDebugView == 2) {
        FragColor = vec4(vec3(roughness), 1.0);
        return;
    }
    if (uDebugView == 3) {
        FragColor = vec4(vec3(ao), 1.0);
        return;
    }
    if (uDebugView == 4) {
        FragColor = vec4(visualizeNormal(N), 1.0);
        return;
    }
    if (uDebugView == 5) {
        FragColor = vec4(vec3(rotationTex), 1.0);
        return;
    }
    if (uDebugView == 6) {
        FragColor = vec4(vec3(strengthTex), 1.0);
        return;
    }

    FragColor = vec4(color, 1.0);
}