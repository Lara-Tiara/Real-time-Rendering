import argparse
from pathlib import Path
import cv2
import numpy as np

MATERIALS = ["crepe_satin", "gingham_check", "hessian_230"]

def normalize_percentile(x, p0=2.0, p1=98.0, eps=1e-8):
    lo, hi = np.percentile(x, [p0, p1])
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)

def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img.astype(np.float32) / 255.0

def save_rgb(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def save_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img8)

def resize_img(img, size):
    if size <= 0:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def normalize01(x, eps=1e-8):
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + eps)

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def decode_normal_gl(normal_rgb):
    n = normal_rgb * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.clip(norm, 1e-8, None)
    return n

def make_guidance(diffuse_rgb, normal_rgb):
    diffuse_lin = srgb_to_linear(diffuse_rgb)
    luma = 0.2126 * diffuse_lin[..., 0] + 0.7152 * diffuse_lin[..., 1] + 0.0722 * diffuse_lin[..., 2]
    luma = normalize01(luma)

    n = decode_normal_gl(normal_rgb)
    nx = n[..., 0]
    ny = n[..., 1]
    nz = n[..., 2]

    normal_detail = np.sqrt(nx * nx + ny * ny)
    normal_detail = normalize01(normal_detail)

    luma_u8 = np.clip(luma * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    luma_eq = clahe.apply(luma_u8).astype(np.float32) / 255.0

    guide = 0.65 * luma_eq + 0.35 * normal_detail
    guide = cv2.GaussianBlur(guide, (0, 0), 0.3)

    return guide, normal_detail, nz

def estimate_direction_and_coherence(guide, sigma):
    gx = cv2.Scharr(guide, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(guide, cv2.CV_32F, 0, 1)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)

    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-8)
    theta_line = theta_grad + 0.5 * np.pi

    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-8)
    coherence = np.clip(coherence, 0.0, 1.0)

    return theta_line, coherence
"""
def build_strength_map(coherence, roughness, normal_detail, strength_scale, strength_gamma):
    rough_term = 1.0 - np.clip(roughness, 0.0, 1.0)
    base = 0.8 * coherence + 0.2 * normal_detail
    strength = base * np.power(rough_term, 0.8)
    strength *= strength_scale
    strength = smoothstep(0.12, 0.85, strength)
    strength = np.power(np.clip(strength, 0.0, 1.0), strength_gamma)
    strength = cv2.GaussianBlur(strength, (0, 0), 1.0)
    return np.clip(strength, 0.0, 1.0)
"""

def build_strength_map(coherence, roughness, normal_detail, strength_scale, strength_gamma, mode="general", invert=False, sharp=False):
    edge = 0.7 * coherence + 0.3 * normal_detail
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8 if sharp else 1.5)
    edge = normalize_percentile(edge, 5.0, 95.0)

    if mode == "woven":
        edge = cv2.GaussianBlur(edge, (0, 0), 1.5 if sharp else 5.0)
    elif mode == "openweave":
        edge = cv2.GaussianBlur(edge, (0, 0), 1.2 if sharp else 4.0)
    else:
        edge = cv2.GaussianBlur(edge, (0, 0), 1.0 if sharp else 2.5)

    strength = 1.0 - edge

    rough_term = 0.55 + 0.45 * (1.0 - np.clip(roughness, 0.0, 1.0))
    strength = strength * rough_term

    strength = normalize_percentile(strength, 5.0, 95.0)
    strength = 0.20 + 0.60 * strength
    strength = np.clip(strength * strength_scale, 0.0, 1.0)
    strength = np.power(strength, strength_gamma)
    strength = cv2.GaussianBlur(strength, (0, 0), 0.5 if sharp else 2.0)

    if invert:
        strength = 1.0 - strength

    return np.clip(strength, 0.0, 1.0)

def encode_direction_map(theta):
    dir2x = np.cos(2.0 * theta)
    dir2y = np.sin(2.0 * theta)
    out = np.zeros((theta.shape[0], theta.shape[1], 3), dtype=np.float32)
    out[..., 0] = dir2x * 0.5 + 0.5
    out[..., 1] = dir2y * 0.5 + 0.5
    out[..., 2] = 0.5
    return out

def encode_rotation_gray(theta):
    rot = np.mod(theta, np.pi) / np.pi
    return np.clip(rot, 0.0, 1.0)

def make_direction_vis(theta, strength):
    hue = (np.mod(theta, np.pi) / np.pi) * 179.0
    hsv = np.zeros((theta.shape[0], theta.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = hue
    hsv[..., 1] = np.clip(np.maximum(strength, 0.2), 0.0, 1.0) * 255.0
    hsv[..., 2] = 255.0
    hsv8 = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb

def make_debug_vis(guide, dir_vis, strength):
    g = np.repeat(guide[..., None], 3, axis=2)
    s = np.repeat(strength[..., None], 3, axis=2)
    return np.concatenate([g, dir_vis, s], axis=1)

def find_file(folder, suffix):
    matches = list(folder.glob(f"*{suffix}"))
    if not matches:
        return None
    return matches[0]

def process_material(folder, size, sigma, strength_scale, strength_gamma):
    diff_path = find_file(folder, "_diff_4k.png")
    normal_path = find_file(folder, "_nor_gl_4k.png")
    rough_path = find_file(folder, "_rough_4k.png")
    ref_rot_path = find_file(folder, "_anisotropy_rotation_4k.png")
    ref_str_path = find_file(folder, "_anisotropy_strength_4k.png")
    mask_path = find_file(folder, "_mask_4k.png")

    if diff_path is None or normal_path is None or rough_path is None:
        raise RuntimeError(f"Missing required maps in {folder}")

    diffuse = resize_img(load_rgb(diff_path), size)
    normal = resize_img(load_rgb(normal_path), size)
    roughness = resize_img(load_gray(rough_path), size)

    guide, normal_detail, nz = make_guidance(diffuse, normal)
    theta, coherence = estimate_direction_and_coherence(guide, sigma)
    #strength = build_strength_map(coherence, roughness, normal_detail, strength_scale, strength_gamma)
    mode = "general"
    invert_strength = True

    if folder.name == "gingham_check":
        mode = "woven"
    elif folder.name == "hessian_230":
        mode = "openweave"

    strength = build_strength_map(
        coherence,
        roughness,
        normal_detail,
        strength_scale,
        strength_gamma,
        mode=mode,
        invert=invert_strength,
        sharp=True
    )

    direction_map = encode_direction_map(theta)
    rotation_gray = encode_rotation_gray(theta)
    dir_vis = make_direction_vis(theta, strength)
    debug_vis = make_debug_vis(guide, dir_vis, strength)

    outdir = folder / "generated"
    save_rgb(outdir / "direction_map.png", direction_map)
    save_gray(outdir / "strength_map.png", strength)
    save_gray(outdir / "reconstructed_anisotropy_rotation.png", rotation_gray)
    save_gray(outdir / "reconstructed_anisotropy_strength.png", strength)
    save_rgb(outdir / "debug_vis.png", debug_vis)

    if ref_rot_path is not None:
        ref_rot = resize_img(load_gray(ref_rot_path), size)
        save_gray(outdir / "reference_anisotropy_rotation.png", ref_rot)

    if ref_str_path is not None:
        ref_str = resize_img(load_gray(ref_str_path), size)
        save_gray(outdir / "reference_anisotropy_strength.png", ref_str)

    if mask_path is not None:
        mask = resize_img(load_gray(mask_path), size)
        save_gray(outdir / "reference_mask.png", mask)

    print(f"[OK] {folder.name}")
    print(f"  {outdir / 'direction_map.png'}")
    print(f"  {outdir / 'strength_map.png'}")
    print(f"  {outdir / 'reconstructed_anisotropy_rotation.png'}")
    print(f"  {outdir / 'reconstructed_anisotropy_strength.png'}")
    print(f"  {outdir / 'debug_vis.png'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth-root", type=str, default=r"..\assets\textures\cloth")
    parser.add_argument("--material", type=str, default="all")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--strength-scale", type=float, default=1.0)
    parser.add_argument("--strength-gamma", type=float, default=1.2)
    args = parser.parse_args()

    cloth_root = Path(args.cloth_root)

    if args.material == "all":
        names = MATERIALS
    else:
        names = [args.material]

    for name in names:
        folder = cloth_root / name
        if not folder.exists():
            print(f"[Skip] not found: {folder}")
            continue
        process_material(folder, args.size, args.sigma, args.strength_scale, args.strength_gamma)

if __name__ == "__main__":
    main()