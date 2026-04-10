import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ═══════════════════════════════════════════════════════════════════════════════
#  VLE DATA  (MeOH/IPA, Wilson + Antoine, 1 atm)
# ═══════════════════════════════════════════════════════════════════════════════
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
vle_interp = interp1d(x_vle, y_vle, kind='cubic',
                      bounds_error=False, fill_value=(0.0, 0.998021))

# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
R        = 5
N_TRAYS  = 11
EMV_TRAY = 0.50    # Murphree efficiency for the 10 column trays
EMV_REB  = 0.686   # Murphree efficiency for the reboiler (from lab data)
N_ACTUAL = 11      # 10 trays + 1 reboiler
EMV      = EMV_TRAY  # alias used in N_THEOR calc
EFF      = EMV_TRAY  # overall tray efficiency for reporting
N_THEOR  = int(round(N_TRAYS * EFF))

rho_MeOH = 0.791   # g/mL at 20°C
rho_IPA  = 0.786
MW_MeOH  = 32.04
MW_IPA   = 60.10

V0_mL  = 850.0
x0_mol = 0.20      # initial still-pot mole fraction MeOH

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def op_line_y(x, xD, R):
    return (R / (R + 1)) * x + xD / (R + 1)

def mol_frac_from_vol_frac(phi):
    n_MeOH = phi / MW_MeOH
    n_IPA  = (1 - phi) / MW_IPA
    return n_MeOH / (n_MeOH + n_IPA)

def moles_from_volume(V_mL, x_mol):
    """Convert volume (mL) of mixture at mol frac x_mol to total moles."""
    phi = brentq(lambda p: mol_frac_from_vol_frac(p) - x_mol, 1e-6, 1-1e-6)
    rho = phi * rho_MeOH + (1 - phi) * rho_IPA
    mass = V_mL * rho
    n_MeOH = (phi * V_mL * rho_MeOH) / MW_MeOH
    n_IPA  = ((1 - phi) * V_mL * rho_IPA) / MW_IPA
    return n_MeOH + n_IPA, phi

def volume_from_moles(F_mol, x_mol):
    """Convert moles at mol frac x_mol back to volume (mL)."""
    phi = brentq(lambda p: mol_frac_from_vol_frac(p) - x_mol, 1e-6, 1-1e-6)
    rho = phi * rho_MeOH + (1 - phi) * rho_IPA
    MW_avg = x_mol * MW_MeOH + (1 - x_mol) * MW_IPA
    return F_mol * MW_avg / rho

def xD_from_xW(xW, R, n_actual=N_ACTUAL, emv_tray=EMV_TRAY, emv_reb=EMV_REB):
    """
    Step n_actual stages UP from xW using the R=5 operating line.
    Stages 1..(n_actual-1) use EMV_TRAY; stage n_actual (reboiler) uses EMV_REB.
    Bisection on xD until the staircase bottom lands at xW.
    """
    x_dense = np.linspace(0, 1, 8000)
    y_dense = vle_interp(x_dense)

    def bottom_of_staircase(xD_try):
        y_op_dense    = op_line_y(x_dense, xD_try, R)
        y_pseudo_tray = y_op_dense + emv_tray * (y_dense - y_op_dense)
        y_pseudo_reb  = y_op_dense + emv_reb  * (y_dense - y_op_dense)
        y_cur = xD_try
        x_bottom = xD_try
        for step in range(n_actual):
            is_reboiler = (step == n_actual - 1)
            y_pseudo = y_pseudo_reb if is_reboiler else y_pseudo_tray
            diffs = y_pseudo - y_cur
            sc = np.where(np.diff(np.sign(diffs)))[0]
            if len(sc) == 0:
                return 0.0
            idx = sc[-1]
            x_lo, x_hi = x_dense[idx], x_dense[idx+1]
            f_lo, f_hi = diffs[idx], diffs[idx+1]
            x_bottom = x_lo - f_lo*(x_hi-x_lo)/(f_hi-f_lo)
            if step == n_actual - 1:
                return x_bottom
            y_cur = op_line_y(x_bottom, xD_try, R)
        return x_bottom

    lo, hi = xW + 1e-5, 0.9995
    for _ in range(80):
        mid = (lo + hi) / 2
        bot = bottom_of_staircase(mid)
        if bot > xW:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

# ═══════════════════════════════════════════════════════════════════════════════
#  FIND STOPPING COMPOSITION
#  The distillation should stop when the instantaneous distillate xD is no
#  longer enriched above the still pot — i.e. when xD(xW) ≈ xW.
#  For MeOH/IPA this happens near xW → 0 (pure IPA limit).
#  A practical stopping criterion: stop when xD drops below some minimum
#  useful purity, OR when all MeOH in the still pot is exhausted.
#
#  We find the minimum xW for which the column can still produce xD > xW
#  by scanning: below a certain xW the operating line pinches the equil curve.
# ═══════════════════════════════════════════════════════════════════════════════
print("Scanning xD(xW) to find practical stop point...")
xW_scan = np.linspace(0.001, x0_mol - 0.001, 500)
xD_scan = np.array([xD_from_xW(xw, R) for xw in xW_scan])

# Stop when xD - xW becomes negligibly small (< 0.005) — column no longer separating
separation = xD_scan - xW_scan
# Find where separation first drops below threshold coming from x0 downward
threshold = 0.005
stop_idx = np.where(separation < threshold)[0]
if len(stop_idx) > 0:
    xW_stop = xW_scan[stop_idx[0]]
else:
    xW_stop = xW_scan[-1]  # fallback: lowest scanned value

print(f"Column stops separating at xW ≈ {xW_stop:.4f}  (xD - xW < {threshold})")

# ═══════════════════════════════════════════════════════════════════════════════
#  RAYLEIGH INTEGRATION
#  ln(F0/F) = integral from x0 down to xW_f  of  dx / (xD(x) - x)
#  Integrate from x0 = 0.20 DOWN to xW_stop
# ═══════════════════════════════════════════════════════════════════════════════
N_POINTS = 400
xW_arr = np.linspace(x0_mol, xW_stop, N_POINTS)
print("Building xD(xW) grid for Rayleigh integration...")
xD_arr = np.array([xD_from_xW(xw, R) for xw in xW_arr])
print("Done.")

integrand = 1.0 / (xD_arr - xW_arr)

# Cumulative integral (xW decreases so dx < 0; negate to get positive value)
ln_F0_over_F = np.zeros(N_POINTS)
for i in range(1, N_POINTS):
    seg = np.trapezoid(integrand[:i+1], xW_arr[:i+1])
    ln_F0_over_F[i] = -seg   # negate: xW_arr is decreasing so trapz gives negative

F0, phi0 = moles_from_volume(V0_mL, x0_mol)
F_arr    = F0 * np.exp(-ln_F0_over_F)
D_arr    = F0 - F_arr

# Average distillate composition by material balance: F0*x0 = F*xW + D*xD_avg
xD_avg_arr = np.where(D_arr > 1e-9,
                      (F0*x0_mol - F_arr*xW_arr) / D_arr,
                      xD_arr)

# ── Final values at stop point ────────────────────────────────────────────────
xW_final  = xW_arr[-1]
F_final   = F_arr[-1]
D_total   = D_arr[-1]
xD_avg    = xD_avg_arr[-1]

V_distillate = volume_from_moles(D_total, xD_avg)
V_stillpot   = volume_from_moles(F_final, xW_final)

print("\n" + "="*62)
print("  INITIAL CHARGE")
print("="*62)
print(f"  Volume                        : {V0_mL:.1f} mL")
print(f"  Mol frac MeOH  x0             : {x0_mol:.4f}")
print(f"  Vol frac MeOH  phi0           : {phi0:.4f}")
print(f"  Total moles    F0             : {F0:.4f} mol")
print(f"  N_theoretical (EFF={EFF*100:.0f}%, R={R})  : {N_THEOR}")
print("\n" + "="*62)
print("  RAYLEIGH PREDICTIONS")
print("="*62)
print(f"  Still-pot final comp   xW,f   : {xW_final:.4f}  mol frac MeOH")
print(f"  Still-pot final moles  F_f    : {F_final:.4f}  mol")
print(f"  Still-pot final volume        : {V_stillpot:.1f}  mL")
print(f"  Distillate collected   D      : {D_total:.4f}  mol")
print(f"  Distillate volume             : {V_distillate:.1f}  mL")
print(f"  Avg distillate comp    xD_avg : {xD_avg:.4f}  mol frac MeOH")
print("="*62)

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE STEPPING — Murphree pseudo-curve + R=5 operating line (for plotting)
#  Identical logic to xD_from_xW so the drawn staircase matches the integration.
# ═══════════════════════════════════════════════════════════════════════════════
def step_stages(xD, xW, R, emv_tray=EMV_TRAY, emv_reb=EMV_REB, n_actual=N_ACTUAL):
    x_dense    = np.linspace(0, 1, 8000)
    y_dense    = vle_interp(x_dense)
    y_op_dense = op_line_y(x_dense, xD, R)
    y_pseudo_tray = y_op_dense + emv_tray * (y_dense - y_op_dense)
    y_pseudo_reb  = y_op_dense + emv_reb  * (y_dense - y_op_dense)

    xs = [xD]; ys = [xD]
    y_cur = xD
    n_stages = 0

    for stage in range(n_actual):
        is_reboiler = (stage == n_actual - 1)
        y_pseudo = y_pseudo_reb if is_reboiler else y_pseudo_tray
        diffs = y_pseudo - y_cur
        sc = np.where(np.diff(np.sign(diffs)))[0]
        if len(sc) == 0: break
        idx = sc[-1]
        x_lo, x_hi = x_dense[idx], x_dense[idx+1]
        f_lo, f_hi = diffs[idx], diffs[idx+1]
        x_eq = x_lo - f_lo*(x_hi-x_lo)/(f_hi-f_lo)
        xs.append(x_eq); ys.append(y_cur)
        n_stages += 1
        if x_eq <= xW: break
        y_op = op_line_y(x_eq, xD, R)
        xs.append(x_eq); ys.append(y_op)
        y_cur = y_op
        if y_cur <= op_line_y(xW, xD, R) or x_eq <= xW: break
    return xs, ys, n_stages

def fmt_ax(ax):
    ax.set_facecolor('white')
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=10, direction='in')
    ax.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
    ax.tick_params(top=True, right=True, which='both')
    ax.grid(False)
    ax.spines[['top','right','bottom','left']].set_linewidth(1.0)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT A — McCabe-Thiele at R=5 showing initial and final operating lines
# ═══════════════════════════════════════════════════════════════════════════════
xD_initial = xD_from_xW(x0_mol, R)
xD_final   = xD_arr[-1]   # distillate when still pot is at xW_stop

sx_i, sy_i, n_st_i = step_stages(xD_initial, x0_mol, R)
sx_f, sy_f, n_st_f = step_stages(xD_final,   xW_final, R)

fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.patch.set_facecolor('white')

ax1.plot(x_vle, y_vle, 'k-', lw=2.2, label='Equilibrium curve', zorder=3)
ax1.plot([0,1], [0,1], 'k--', lw=1.1, alpha=0.45, label='y = x', zorder=2)

# Initial op line and stages
x_op_i = np.linspace(x0_mol, xD_initial, 300)
ax1.plot(x_op_i, op_line_y(x_op_i, xD_initial, R),
         color='#2166ac', lw=1.8, label='_nolegend_')
ax1.plot(sx_i, sy_i, color='#2166ac', lw=1.5, linestyle='--', alpha=0.85,
         label='_nolegend_')

# Final op line and stages
x_op_f = np.linspace(xW_final, xD_final, 300)
ax1.plot(x_op_f, op_line_y(x_op_f, xD_final, R),
         color='#d6604d', lw=1.8, label='_nolegend_')
ax1.plot(sx_f, sy_f, color='#d6604d', lw=1.5, linestyle='--', alpha=0.85,
         label='_nolegend_')

# Key points
ax1.scatter([xD_initial], [xD_initial], color='#2166ac', s=75, zorder=6)
ax1.scatter([x0_mol],     [x0_mol],     color='#2166ac', s=75, marker='s', zorder=6)
ax1.scatter([xD_final],   [xD_final],   color='#d6604d', s=75, zorder=6)
ax1.scatter([xW_final],   [xW_final],   color='#d6604d', s=75, marker='s', zorder=6)

ax1.annotate(f'$x_D^0$={xD_initial:.3f}', xy=(xD_initial,xD_initial),
             xytext=(xD_initial-0.06, xD_initial+0.06), fontsize=8.5, color='#2166ac', ha='right',
             arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8))
ax1.annotate(f'$x_0$={x0_mol:.3f}', xy=(x0_mol,x0_mol),
             xytext=(x0_mol+0.05, x0_mol-0.08), fontsize=8.5, color='#2166ac', ha='left',
             arrowprops=dict(arrowstyle='->', color='#2166ac', lw=0.8))
ax1.annotate(f'$x_D^f$={xD_final:.3f}', xy=(xD_final,xD_final),
             xytext=(xD_final-0.06, xD_final+0.06), fontsize=8.5, color='#d6604d', ha='right',
             arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))
ax1.annotate(f'$x_{{W,f}}$={xW_final:.3f}', xy=(xW_final,xW_final),
             xytext=(xW_final+0.05, xW_final-0.08), fontsize=8.5, color='#d6604d', ha='left',
             arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))

ax1.set_xlim(0,1); ax1.set_ylim(0,1)
ax1.set_xlabel('$x_{MeOH}$ (liquid mole fraction)', fontsize=12)
ax1.set_ylabel('$y_{MeOH}$ (vapor mole fraction)', fontsize=12)
ax1.xaxis.set_major_locator(MultipleLocator(0.2))
ax1.yaxis.set_major_locator(MultipleLocator(0.2))
fmt_ax(ax1)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax1.text(0.04, 0.96, 'A', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/rayleigh_mccabe_thiele.png',
            dpi=1200, bbox_inches='tight', facecolor='white')
print("Saved: rayleigh_mccabe_thiele.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT B — Rayleigh integrand
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 5))
fig2.patch.set_facecolor('white')

ax2.plot(xW_arr, integrand, color='#2166ac', lw=2.0)
ax2.fill_between(xW_arr, integrand, alpha=0.15, color='#2166ac',
                 label=f'Area = ln($F_0/F_f$) = {ln_F0_over_F[-1]:.4f}')

ax2.set_xlabel('$x_W$ (still-pot mole fraction MeOH)', fontsize=12)
ax2.set_ylabel(r'$1\,/\,(x_D^* - x_W)$', fontsize=12)
ax2.invert_xaxis()
ax2.xaxis.set_major_locator(MultipleLocator(0.05))
ax2.xaxis.set_minor_locator(MultipleLocator(0.01))
ax2.yaxis.set_major_locator(MultipleLocator(5))
ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=10, direction='in')
ax2.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
ax2.tick_params(top=True, right=True, which='both')
ax2.grid(False)
ax2.set_facecolor('white')
ax2.spines[['top','right','bottom','left']].set_linewidth(1.0)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.text(0.04, 0.96, 'B', transform=ax2.transAxes,
         fontsize=16, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/rayleigh_integrand.png',
            dpi=1200, bbox_inches='tight', facecolor='white')
print("Saved: rayleigh_integrand.png")

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS C & D — Composition profiles and distillate collected
# ═══════════════════════════════════════════════════════════════════════════════
frac_dist = D_arr / F0

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))
fig3.patch.set_facecolor('white')

# C: compositions vs fraction distilled
ax3a.plot(frac_dist, xD_arr,     color='#2166ac', lw=2.0, label='Instantaneous $x_D$')
ax3a.plot(frac_dist, xW_arr,     color='#d6604d', lw=2.0, label='Still-pot $x_W$')
ax3a.plot(frac_dist, xD_avg_arr, color='#4dac26', lw=2.0, linestyle='--',
          label='Cumulative avg $\\bar{x}_D$')

ax3a.scatter([frac_dist[-1]], [xD_avg],  color='#4dac26', s=75, zorder=5)
ax3a.scatter([frac_dist[-1]], [xW_final],color='#d6604d', s=75, marker='s', zorder=5)
ax3a.annotate(f'$\\bar{{x}}_D$ = {xD_avg:.3f}',
              xy=(frac_dist[-1], xD_avg),
              xytext=(frac_dist[-1]-0.06, xD_avg+0.06),
              fontsize=8.5, color='#4dac26', ha='right',
              arrowprops=dict(arrowstyle='->', color='#4dac26', lw=0.8))
ax3a.annotate(f'$x_{{W,f}}$ = {xW_final:.3f}',
              xy=(frac_dist[-1], xW_final),
              xytext=(frac_dist[-1]-0.06, xW_final-0.07),
              fontsize=8.5, color='#d6604d', ha='right',
              arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))

ax3a.set_xlim(0, max(frac_dist)*1.05)
ax3a.set_ylim(0, 1)
ax3a.set_xlabel('Fraction of charge distilled  ($D/F_0$)', fontsize=11)
ax3a.set_ylabel('Mole fraction MeOH', fontsize=11)
ax3a.xaxis.set_major_locator(MultipleLocator(0.1))
ax3a.xaxis.set_minor_locator(MultipleLocator(0.025))
ax3a.yaxis.set_major_locator(MultipleLocator(0.2))
ax3a.yaxis.set_minor_locator(MultipleLocator(0.05))
ax3a.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=9, direction='in')
ax3a.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
ax3a.tick_params(top=True, right=True, which='both')
ax3a.grid(False)
ax3a.set_facecolor('white')
ax3a.spines[['top','right','bottom','left']].set_linewidth(1.0)
ax3a.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax3a.text(0.04, 0.96, 'C', transform=ax3a.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')

# D: distillate moles collected vs xW
ax3b.plot(xW_arr, D_arr, color='#7b2d8b', lw=2.0)
ax3b.scatter([xW_final], [D_total], color='#7b2d8b', s=75, zorder=5,
             label=f'$D_{{total}}$ = {D_total:.3f} mol  ({V_distillate:.1f} mL)\n'
                   f'$x_{{W,f}}$ = {xW_final:.4f}')
ax3b.invert_xaxis()
ax3b.set_xlabel('Still-pot composition $x_W$  (mol frac MeOH)', fontsize=11)
ax3b.set_ylabel('Cumulative distillate collected  (mol)', fontsize=11)
ax3b.xaxis.set_major_locator(MultipleLocator(0.05))
ax3b.xaxis.set_minor_locator(MultipleLocator(0.01))
ax3b.yaxis.set_minor_locator(MultipleLocator(0.1))
ax3b.tick_params(axis='both', which='major', length=6, width=1.0, labelsize=9, direction='in')
ax3b.tick_params(axis='both', which='minor', length=3, width=0.7, direction='in')
ax3b.tick_params(top=True, right=True, which='both')
ax3b.grid(False)
ax3b.set_facecolor('white')
ax3b.spines[['top','right','bottom','left']].set_linewidth(1.0)
ax3b.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax3b.text(0.04, 0.96, 'D', transform=ax3b.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')

plt.tight_layout(w_pad=3.0)
plt.show()
