import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress

# ── DATA ──────────────────────────────────────────────────────────────────────
vol_frac = np.array([0.00, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])
nD       = np.array([1.37718, 1.37271, 1.36509, 1.35779, 1.35015, 1.34331, 1.33624, 1.32958])

# ── LINEAR FIT ────────────────────────────────────────────────────────────────
slope, intercept, r, p, se = linregress(vol_frac, nD)
x_fit = np.linspace(0, 1, 500)
y_fit = slope * x_fit + intercept
r2 = r**2

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Data points
ax.scatter(vol_frac, nD, color='black', s=60, zorder=5, label='Calibration data')

# Fit line
eq_str = f'$n_D = {slope:.4f}\\,x_{{MeOH}} + {intercept:.5f}$\n$R^2 = {r2:.6f}$'
ax.plot(x_fit, y_fit, color='#2166ac', lw=2, zorder=4, label=eq_str)

# ── Axes
ax.set_xlim(-0.02, 1.02)
y_pad = 0.0003
ax.set_ylim(nD.min() - y_pad*6, nD.max() + y_pad*6)

ax.set_xlabel('Volume Fraction MeOH ($x_{MeOH,vol}$)', fontsize=12)
ax.set_ylabel('Refractive Index $n_D$ (20°C)', fontsize=12)

# ── Ticks
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.005))
ax.yaxis.set_minor_locator(MultipleLocator(0.001))

ax.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=10, direction='in')
ax.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
ax.tick_params(top=True, right=True, which='both')

ax.spines[['top', 'right', 'bottom', 'left']].set_linewidth(1.0)
ax.grid(False)

ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()
