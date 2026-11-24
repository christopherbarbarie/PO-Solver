import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from roughSynth import fractal
from roughSynth.facets import gen_normals
from po_circle import fresnel, fresnel_zone, flat_disk
import sys
sys.path.insert(0, '/Users/chba2870/Desktop/Work/fractal_sim')
from model import disk_loss
from sim import grid_floor

COLORS = lambda n: plt.get_cmap('cmr.sapphire')(np.linspace(0.2, 0.92, n))

def discretize_fresnel_disk(h, wavelength, dx):
    a = fresnel_zone(h, wavelength)
    spacing = 2 * int(np.ceil(a / dx))
    x = (np.arange(spacing) - (spacing - 1) / 2) * dx
    X, Y = np.meshgrid(x, x)
    inside = X**2 + Y**2 <= a**2
    return X, Y, inside

def po(x, y, z, n, h, dx, wavelength, eps=np.inf):
    k = 2 * np.pi / wavelength
    r = np.stack([x, y, z - h], -1)
    R = np.linalg.norm(r, axis=-1, keepdims=True)
    i= r / R
    al = np.cross(i, n)
    al /= np.linalg.norm(al, axis=-1, keepdims=True)
    be = np.cross(i, al)
    i_n = np.sum(i * n, -1, keepdims=True)
    Rs, Rp = fresnel(-i_n, eps)
    f = np.exp(-2j * k * R) / R**2 * i_n * (be[:, :1] * Rp * be - al[:, :1] * Rs * al)
    return 1j / (2 * np.pi) * np.sum(f * dx**2 / n[..., 2:], 0)

def coherence_loss(X, Y, x, y, m, h, sigmas, inside, wavelength, eps, H, dx, n):
    fresnel_radius = fresnel_zone(h, wavelength)
    flat_power = abs(flat_disk(h, fresnel_radius, wavelength, eps)[0])**2
    P = []
    for i in range(n):
        field1, field2, x0, y0 = fractal(2*m, 2*m, 2, H, m*dx)
        for z in (field1, field2):
            z = z * (m * dx / wavelength)**H
            P.append([abs(po(x, y, s * z[inside], gen_normals(X, Y, s * z)[0][inside], h, dx, wavelength, eps)[0])**2 for s in sigmas])
    P = np.array(P) / flat_power
    return P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))

HURSTS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
MARKERS = 'os^Dv<>'
db = lambda x: 10 * np.log10(np.maximum(x, 1e-30))

def plot_model_vs_po(files, margin=10.0, below=10.0, smax=0.25):
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False})
    fig, axes = plt.subplots(2, len(files), figsize=(max(3.6 * len(files), 5.4), 4.6), sharex=True, height_ratios=[3, 1], squeeze=False, constrained_layout=True)
    fine = np.linspace(0, smax, 101)
    for i, name in enumerate(files):
        d = np.load(name)
        h, s, ax, dax = float(d['h']), d['sigma'], axes[0, i], axes[1, i]
        floor = db(grid_floor(h, float(d['dx'])))
        for k, (H, P, se, color) in enumerate(zip(d['H'], d['P'], d['se'], COLORS(len(d['H'])))):
            use = db(P) > floor + margin
            ax.plot(fine, db(disk_loss(fine, H, h)), color=color, lw=1.5)
            ax.plot([], [], color=color, lw=1.5, marker=MARKERS[k], ms=3.5, label=f'H = {H}')
            ax.plot(s[use], db(P[use]), MARKERS[k], color=color, ms=3.2)
            dax.errorbar(s[use] + 0.0012 * (k - 3), db(P[use]) - db(disk_loss(s[use], H, h)),
                         [db(P[use]) - db(P[use] - se[use]), db(P[use] + se[use]) - db(P[use])], fmt=MARKERS[k], color=color, ms=2.6, lw=0.8)
        ax.axhline(floor, color='grey', ls=':', label='Grid Floor')
        ax.set_ylim(floor - below, 3)
        ax.text(0.98, 0.95, f'Altitude = {h:g} $\\lambda$', transform=ax.transAxes, ha='right', va='top', fontsize=9)
        dax.axhline(0, color='k', lw=0.8)
        for a in (ax, dax):
            a.grid(alpha=0.3)
    axes[0, 0].set_xlim(-0.005, smax)
    axes[0, 0].set_ylabel('Coherence Loss (dB)')
    axes[1, 0].set_ylabel('Difference\n(dB)')
    fig.align_ylabels(axes[:, 0])
    hs, labels = axes[0, 0].get_legend_handles_labels()
    hs += [plt.Line2D([], [], color='k', lw=1.5), plt.Line2D([], [], color='k', marker='o', ls='', ms=3)]
    fig.legend(hs, labels + ['Analytical Model', 'Physical Optics'], ncol=4, loc='outside upper center', fontsize=8.5)
    fig.supxlabel(r'$\sigma_\lambda$ (RMS height at lag $\lambda$, in $\lambda$)', fontsize=10)
    fig.savefig('model_vs_po.png', dpi=300)

def main():
    wavelength = 10
    dx = wavelength/4
    sigmas = np.linspace(0, 0.35, 36)
    for h, n in ((2000, 50), (32000, 50)):
        np.random.seed(h)
        X, Y, inside = discretize_fresnel_disk(h*wavelength, wavelength, dx)
        x = X[inside]
        y = Y[inside]
        m = X.shape[0]
        P, se = np.array([coherence_loss(X, Y, x, y, m, h*wavelength, sigmas*wavelength, inside, wavelength, 8/3.15, H, dx, n) for H in HURSTS]).transpose(1, 0, 2)
        np.savez(f'po_h{h}.npz', H=HURSTS, sigma=sigmas, P=P, se=se, h=h, dx=dx/wavelength, nsurf=2*n)
    plot_model_vs_po(['po_h2000.npz', 'po_h32000.npz'])

if __name__ == "__main__":
    main()
