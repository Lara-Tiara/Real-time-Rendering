import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fontsize=11):
    r = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=14,
                        linewidth=1.2, color="black")
    ax.add_patch(a)

plt.figure(figsize=(14, 6))
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Row layout
y = 0.62
h = 0.22
w = 0.16
gap = 0.04

x0 = 0.04
box(ax, x0, y, w, h, "Input maps\nDiffuse (sRGB)\nNormal (GL)\nRoughness\nAO")
arrow(ax, x0 + w, y + h/2, x0 + w + gap, y + h/2)

x1 = x0 + w + gap
box(ax, x1, y, w, h, "Guidance\nluma + normal detail\nCLAHE + blur")
arrow(ax, x1 + w, y + h/2, x1 + w + gap, y + h/2)

x2 = x1 + w + gap
box(ax, x2, y, w, h, "Structure tensor\nScharr gx/gy\nblur Jxx/Jyy/Jxy")
arrow(ax, x2 + w, y + h/2, x2 + w + gap, y + h/2)

x3 = x2 + w + gap
box(ax, x3, y, w, h, "Direction θ\nprincipal orientation\n(+π/2)")
arrow(ax, x3 + w, y + h/2, x3 + w + gap, y + h/2)

x4 = x3 + w + gap
box(ax, x4, y, w, h, "Strength\ncoherence + heuristics")

# Outputs row
y2 = 0.18
w2 = 0.22
h2 = 0.18

box(ax, 0.10, y2, w2, h2, "Output\nreconstructed_anisotropy_rotation.png")
box(ax, 0.39, y2, w2, h2, "Output\nreconstructed_anisotropy_strength.png")
box(ax, 0.68, y2, w2, h2, "Debug\n(debug_vis.png)\n(dir/strength preview)")

arrow(ax, x4 + w/2, y, 0.21, y2 + h2)
arrow(ax, x4 + w/2, y, 0.50, y2 + h2)
arrow(ax, x4 + w/2, y, 0.79, y2 + h2)

plt.tight_layout()
plt.savefig("pipeline.png", dpi=200)
print("Saved pipeline.png")