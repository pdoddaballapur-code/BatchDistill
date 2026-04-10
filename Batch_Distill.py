import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator

# ── VLE DATA (MeOH/IPA, Wilson + Antoine, 1 atm) ─────────────────────────────
x_vle = np.array([
    0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
    0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40,
    0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60,
    0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80,
    0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0
])
y_vle = np.array([
    0, 0.035119, 0.06946, 0.103033, 0.135848, 0.167916, 0.199247,
    0.22985, 0.259737, 0.288917, 0.3174, 0.345196, 0.372317, 0.398771,
    0.424569, 0.449721, 0.474238, 0.498129, 0.521405, 0.544075, 0.566149,
    0.587639, 0.608553, 0.628901, 0.648694, 0.667942, 0.686654, 0.70484,
    0.72251, 0.739674, 0.756342, 0.772522, 0.788226, 0.803461, 0.818239,
    0.832568, 0.846458, 0.859918, 0.872958, 0.885587, 0.897814, 0.909649,
    0.9211, 0.932177, 0.942889, 0.953245, 0.963253, 0.972924, 0.982266,
    0.991289, 0.998021
])

vle_interp = interp1d(x_vle, y_vle, kind='cubic', bounds_error=False, fill_value=(0, 0.998021))

# ── EXPERIMENTAL DATA ─────────────────────────────────────────────────────────
runs = [
    {"label": "142 W", "xD": 0.894312, "xW": 0.195261, "x_last_tray": 0.331627, "T_top": 68.2, "T_bot": 81.3, "color": "#2166ac"},
    {"label": "132 W", "xD": 0.760700, "xW": 0.196239, "x_last_tray": 0.349182, "T_top": 70.6, "T_bot": 81.2, "color": "#d6604d"},
    {"label": "137 W", "xD": 0.788608, "xW": 0.196565, "x_last_tray": 0.351111, "T_top": 70.3, "T_bot": 81.3, "color": "#4dac26"},
    {"label": "146 W", "xD": 0.801219, "xW": 0.187727, "x_last_tray": 0.365582, "T_top": 68.5, "T_bot": 81.5, "color": "#7b2d8b"},
]

# ── STAGE STEPPING AT TOTAL REFLUX ───────────────────────────────────────────
def step_stages_total_reflux(xD, xW, vle_interp, n_steps=30):
    x_dense = np.linspace(0, 1, 5000)
    y_dense = vle_interp(x_dense)
    xs = [xD]
    ys = [xD]
    y_current = xD
    n_stages = 0
    for _ in range(n_steps):
        diffs = y_dense - y_current
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) == 0:
            break
        idx = sign_changes[-1]
        x_lo, x_hi = x_dense[idx], x_dense[idx + 1]
        f_lo, f_hi = diffs[idx], diffs[idx + 1]
        x_eq = x_lo - f_lo * (x_hi - x_lo) / (f_hi - f_lo)
        xs.append(x_eq)
        ys.append(y_current)
        n_stages += 1
        if x_eq <= xW:
            break
        xs.append(x_eq)
        ys.append(x_eq)
        y_current = x_eq
        if x_eq <= xW:
            break
    return xs, ys, n_stages

# ── PLOTTING ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
axes = axes.flatten()
fig.patch.set_facecolor('white')

x_diag = np.linspace(0, 1, 200)
panel_labels = ['A', 'B', 'C', 'D']

for ax, run, panel in zip(axes, runs, panel_labels):
    xD = run["xD"]
    xW = run["xW"]
    x_lt = run["x_last_tray"]
    color = run["color"]

    ax.plot(x_vle, y_vle, color='black', lw=2.2, label='Equilibrium curve', zorder=3)
    ax.plot(x_diag, x_diag, 'k--', lw=1.5, alpha=0.7, label='y = x  (total reflux)', zorder=2)

    sx, sy, n_stages = step_stages_total_reflux(xD, xW, vle_interp)
    ax.plot(sx, sy, color=color, lw=1.8, linestyle='-', alpha=0.9,
            label=f'Theoretical stages: {n_stages}', zorder=5)


    ax.scatter([xD],   [xD],   color=color,   s=70, zorder=6)
    ax.scatter([xW],   [xW],   color='black', s=70, zorder=6, marker='s')
    ax.scatter([x_lt], [x_lt], color='gray',  s=65, zorder=6, marker='^')

    ax.annotate(f'$x_D$ = {xD:.3f}', xy=(xD, xD),
                xytext=(xD - 0.06, xD + 0.07), fontsize=8.5, color=color, ha='right',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
    ax.annotate(f'$x_W$ = {xW:.3f}', xy=(xW, xW),
                xytext=(xW + 0.05, xW - 0.09), fontsize=8.5, color='black', ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
    ax.annotate(f'Last tray\n$x$ = {x_lt:.3f}', xy=(x_lt, x_lt),
                xytext=(x_lt + 0.06, x_lt - 0.11), fontsize=7.5, color='gray', ha='left',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Panel label
    ax.text(0.04, 0.96, panel, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor('white')
    ax.set_xlabel('$x_{MeOH}$ (liquid mole fraction)', fontsize=11)
    if ax in [axes[0], axes[2]]:
        ax.set_ylabel('$y_{MeOH}$ (vapor mole fraction)', fontsize=11)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=9, direction='in')
    ax.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
    ax.tick_params(top=True, right=True, which='both')

    ax.grid(False)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.85, bbox_to_anchor=(0.01, 0.88))
    ax.spines[['top', 'right', 'bottom', 'left']].set_linewidth(1.0)

plt.tight_layout(h_pad=3.0, w_pad=2.5)
plt.show()