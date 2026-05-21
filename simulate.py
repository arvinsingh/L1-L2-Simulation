import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon, Circle
from pathlib import Path

sns.set_theme(
    context="talk",
    style="whitegrid",
    palette="deep",
    font_scale=0.95,
)


# unconstrained optimum
w_star = np.array([2.6, 1.15])

# positive definite matrix w/ some correl b/w coords
A = np.array([
    [3.0, 1.1],
    [1.1, 1.0],
])

# constraint radius & optimization params
radius = 1.0
learning_rate = 0.12
n_steps = 100

# outside constraint space
# ||w0||_1 = 2.00, ||w0||_2 ≈ 1.414
w0_outside = np.array([-1.15, -0.85])


def loss(w):
    """Quadratic loss function - L(w) = 0.5 * (w - w_star)^T A (w - w_star)
    Obsolete, vectorized later.
    """
    d = w - w_star
    return 0.5 * d @ A @ d


def grad(w):
    """Gradient of the loss grad L(w) = A (w - w_star)"""
    return (w - w_star) @ A.T

def project_l2_ball(v, r=1.0):
    """Euclidean proj onto the L2 ball."""
    norm = np.linalg.norm(v, 2)
    return v.copy() if norm <= r else (r / norm) * v


def project_l1_diamond(v, r=1.0):
    """Euclidean proj onto the L1 diamond."""
    if np.linalg.norm(v, 1) <= r:
        return v.copy()

    u = np.sort(np.abs(v))[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, len(u) + 1) > (cssv - r)
    rho = np.where(rho_candidates)[0][-1]
    theta = (cssv[rho] - r) / (rho + 1.0)

    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


def run_simulation(projector, start=w0_outside):
    """
    1- Show the raw outside initialization
    2- Project it onto the feasible set
    3- Run projected gradient descent
    """
    projected_start = projector(start, radius)

    path = [start.copy(), projected_start.copy()]
    w = projected_start.copy()

    for _ in range(n_steps):
        raw_update = w - learning_rate * grad(w)
        w = projector(raw_update, radius)
        path.append(w.copy())

    return np.array(path)


path_l1_outside = run_simulation(project_l1_diamond)
path_l2_outside = run_simulation(project_l2_ball)


# grid for loss contours
x = np.linspace(-1.6, 3.05, 540)
y = np.linspace(-1.35, 2.0, 540)
X, Y = np.meshgrid(x, y)

D0 = X - w_star[0]
D1 = Y - w_star[1]

# Z = np.empty_like(X)
# for i in range(X.shape[0]):
#     for j in range(X.shape[1]):
#         Z[i, j] = loss(np.array([X[i, j], Y[i, j]]))
# vectorized below

Z = 0.5 * (
    A[0, 0] * D0**2
    + 2 * A[0, 1] * D0 * D1
    + A[1, 1] * D1**2
)

levels = np.geomspace(max(Z.min() + 0.035, 0.06), np.percentile(Z, 94), 22)

# neg grad field
gx = np.linspace(-1.25, 1.18, 18)
gy = np.linspace(-1.12, 1.12, 18)
GX, GY = np.meshgrid(gx, gy)
U = np.zeros_like(GX)
V = np.zeros_like(GY)

# for i in range(GX.shape[0]):
#     for j in range(GX.shape[1]):
#         direction = -grad(np.array([GX[i, j], GY[i, j]]))
#         norm = np.linalg.norm(direction)
#         if norm > 0:
#             direction = direction / norm
#         U[i, j], V[i, j] = direction
# vectorized below

G = np.stack([GX, GY], axis=-1)
D = -grad(G)
norms = np.linalg.norm(D, axis=-1, keepdims=True)
D_norm = D / np.maximum(norms, 1e-12)
U = D_norm[..., 0]
V = D_norm[..., 1]



def draw_constraint(ax, kind, color):
    if kind == "l1":
        diamond = np.array([
            [0, radius],
            [radius, 0],
            [0, -radius],
            [-radius, 0],
        ])
        ax.add_patch(
            Polygon(
                diamond,
                closed=True,
                facecolor=color,
                edgecolor="black",
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
        )
        ax.add_patch(
            Polygon(
                diamond,
                closed=True,
                fill=False,
                edgecolor="black",
                linewidth=3.5,
                zorder=5,
                label=r"$\|w\|_1 \leq 1$ feasible set",
            )
        )
    else:
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                facecolor=color,
                edgecolor="none",
                alpha=0.12,
                zorder=2,
            )
        )
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                fill=False,
                edgecolor="black",
                linewidth=3.5,
                zorder=5,
                label=r"$\|w\|_2 \leq 1$ feasible set",
            )
        )


def plot_outside_init(ax, path, kind, title, color):
    # loss landscape
    ax.contourf(X, Y, Z, levels=levels, cmap="mako_r", alpha=0.28)
    contours = ax.contour(X, Y, Z, levels=levels, cmap="mako", linewidths=1.05, alpha=0.88)
    ax.clabel(contours, contours.levels[::4], inline=True, fontsize=8, fmt="%.1f")

    # feasible set
    draw_constraint(ax, kind, color)

    # neg grad field
    ax.quiver(
        GX,
        GY,
        U,
        V,
        color="dimgray",
        alpha=0.42,
        width=0.0022,
        scale=15,
        zorder=3,
    )

    # path segments
    outside = path[0]
    projected = path[1]
    optimization_path = path[1:]

    # first proj styled differently
    ax.plot(
        [outside[0], projected[0]],
        [outside[1], projected[1]],
        linestyle="--",
        linewidth=2.4,
        color="crimson",
        alpha=0.9,
        label="initial projection",
        zorder=7,
    )
    ax.annotate(
        "",
        xy=projected,
        xytext=outside,
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="crimson", mutation_scale=17),
        zorder=8,
    )

    # optim traj after proj
    ax.plot(
        optimization_path[:, 0],
        optimization_path[:, 1],
        color=color,
        linewidth=3.0,
        zorder=7,
        label="projected GD trajectory",
    )

    sizes = np.linspace(38, 115, len(optimization_path))
    ax.scatter(
        optimization_path[:, 0],
        optimization_path[:, 1],
        s=sizes,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        alpha=0.84,
        zorder=8,
    )

    # arrows along optim traj
    for idx in [2, 5, 10, 20, 40, 65]:
        if idx + 1 < len(optimization_path):
            p = optimization_path[idx]
            q = optimization_path[idx + 1]
            if np.linalg.norm(q - p) > 1e-8:
                ax.annotate(
                    "",
                    xy=q,
                    xytext=p,
                    arrowprops=dict(arrowstyle="-|>", lw=1.35, color=color, mutation_scale=14),
                    zorder=9,
                )

    # all special pts
    ax.scatter(
        outside[0],
        outside[1],
        marker="P",
        s=230,
        color="crimson",
        edgecolor="white",
        linewidth=1.0,
        label="outside initialization",
        zorder=10,
    )
    ax.scatter(
        projected[0],
        projected[1],
        marker="s",
        s=160,
        color="gold",
        edgecolor="black",
        linewidth=1.0,
        label="projected start",
        zorder=10,
    )
    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        marker="X",
        s=210,
        color="black",
        edgecolor="white",
        linewidth=0.9,
        label="final solution",
        zorder=11,
    )
    ax.scatter(
        w_star[0],
        w_star[1],
        marker="*",
        s=340,
        color="darkorange",
        edgecolor="white",
        linewidth=0.7,
        label="unconstrained optimum",
        zorder=10,
    )

    # annotate with note
    final = path[-1]
    if kind == "l1":
        note = (
            "L1 projection pulls to a facet,\n"
            "then the loss pushes the path\n"
            "toward a sparse corner."
        )
        note_xytext = (1.35, 0.50)
    else:
        note = (
            "L2 projection lands on the circle;\n"
            "the smooth boundary keeps both\n"
            "coordinates active."
        )
        note_xytext = (1.28, 0.50)

    ax.annotate(
        note,
        xy=final,
        xytext=note_xytext,
        arrowprops=dict(arrowstyle="->", lw=1.4, color="black"),
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.78),
        zorder=12,
    )

    ax.text(
        0.025,
        0.035,
        f"outside init = ({outside[0]:.2f}, {outside[1]:.2f})\n"
        f"projected start = ({projected[0]:.3f}, {projected[1]:.3f})\n"
        f"final = ({final[0]:.3f}, {final[1]:.3f})",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="gray", alpha=0.82),
        zorder=12,
    )

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.32)
    ax.axvline(0, color="black", linewidth=0.9, alpha=0.32)
    ax.set_xlim(-1.45, 2.88)
    ax.set_ylim(-1.25, 1.78)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"weight $w_1$")
    ax.set_ylabel(r"weight $w_2$")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, framealpha=0.92)


# build plot
out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)
outside_l1_file = out_dir / "l1.png"
outside_l2_file = out_dir / "l2.png"
outside_combo_file = out_dir / "l1_l2.png"

palette = sns.color_palette("deep")
l1_color = palette[0]
l2_color = palette[2]

fig, ax = plt.subplots(figsize=(9.5, 6.9))
plot_outside_init(
    ax,
    path_l1_outside,
    "l1",
    "L1 projected gradient descent initialized outside the constraint",
    l1_color,
)
fig.tight_layout()
fig.savefig(outside_l1_file, dpi=260, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(9.5, 6.9))
plot_outside_init(
    ax,
    path_l2_outside,
    "l2",
    "L2 projected gradient descent initialized outside the constraint",
    l2_color,
)
fig.tight_layout()
fig.savefig(outside_l2_file, dpi=260, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(17.7, 6.6), sharex=True, sharey=True)
plot_outside_init(
    axes[0],
    path_l1_outside,
    "l1",
    "L1: outside initialization",
    l1_color,
)
plot_outside_init(
    axes[1],
    path_l2_outside,
    "l2",
    "L2: outside initialization",
    l2_color,
)
fig.suptitle(
    "Projected gradient descent from an infeasible starting point",
    y=1.03,
    fontsize=21,
)
fig.tight_layout()
fig.savefig(outside_combo_file, dpi=260, bbox_inches="tight")
plt.show()

print("Outside initialization norms:")

print(f'w0_outside: ({w0_outside[0]:.2f}, {w0_outside[1]:.2f})')
print(f'L1 norm: {np.linalg.norm(w0_outside, 1):.2f}')
print(f'L2 norm: {np.linalg.norm(w0_outside, 2):.2f}')

print(f'L1 projected start: ({path_l1_outside[1][0]:.3f}, {path_l1_outside[1][1]:.3f})')
print(f'L1 final: ({path_l1_outside[-1][0]:.3f}, {path_l1_outside[-1][1]:.3f})')

print(f'L2 projected start: ({path_l2_outside[1][0]:.3f}, {path_l2_outside[1][1]:.3f})')
print(f'L2 final: ({path_l2_outside[-1][0]:.3f}, {path_l2_outside[-1][1]:.3f})')

print(outside_l1_file)
print(outside_l2_file)
print(outside_combo_file)
print("Plots saved to:", out_dir.resolve())
