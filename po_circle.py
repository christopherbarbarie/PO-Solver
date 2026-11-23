import numpy as np
import matplotlib.pyplot as plt

def fresnel_zone(h, wavelength):
    return np.sqrt((wavelength * h) / 2)

def fresnel(cos1, eps):
    Z2, cos2 = 1 / np.sqrt(eps), np.sqrt(1 - (1 - cos1**2) / eps)
    return (Z2 * cos1 - cos2) / (Z2 * cos1 + cos2), (cos1 - Z2 * cos2) / (Z2 * cos2 + cos1)

def flat_disk(h, a, wavelength, eps=np.inf, n=400):
    k = 2 * np.pi / wavelength
    rho, phi = np.meshgrid(np.linspace(0, a, n), np.linspace(0, 2 * np.pi, n, endpoint=False))
    r = np.stack([rho * np.cos(phi), rho * np.sin(phi), -h * np.ones_like(rho)], -1)
    R = np.linalg.norm(r, axis=-1, keepdims=True)
    i, nh, x = r / R, np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
    al = np.cross(i, nh)
    al /= np.maximum(np.linalg.norm(al, axis=-1, keepdims=True), 1e-300)
    be = np.cross(i, al)
    Rs, Rp = fresnel(-i @ nh, eps)
    f = np.exp(-2j * k * R) / R**2 * (i @ nh)[..., None] * ((be @ x * Rp)[..., None] * be - (al @ x * Rs)[..., None] * al)
    return 1j / (2 * np.pi) * np.trapezoid(f.mean(0) * 2 * np.pi * rho[0][:, None], rho[0], axis=0)

def ratio(altitudes, wavelength):
    ratios = []
    for h in altitudes:
        a = fresnel_zone(h, wavelength)
        E = flat_disk(h, a, wavelength)
        I = 2 * np.pi * h**2 * abs(E[0]) 
        ratios.append(I**2/(h * wavelength)**2)
    return ratios

def plot_ratio(altitudes, ratios, wavelength):
    plt.style.use('paper')
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    ax.axhline(1, color='0.5', ls='--', lw=1.5)
    #ax.text(0.98, 1.0, 'Asymptote', color='0.4', ha='right', va='bottom',
    #        transform=ax.get_yaxis_transform())
    x = np.asarray(altitudes) / wavelength
    ax.plot(x, ratios, color='#1f4e79', lw=3)
    ax.set_xlim(x[0], x[-1])
    pad = 0.08 * (1 - min(ratios))
    ax.set_ylim(min(ratios) - pad, 1 + pad)
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.set_xlabel(r'Altitude ($h/\lambda$)')
    ax.set_ylabel(r'Normalised Phase Integral Power ($I^2/(h\lambda)^2$)')
    return fig, ax

def main():
    wavelength = 10
    altitudes = np.linspace(2000*wavelength, 32000*wavelength, 200)
    ratios = ratio(altitudes, wavelength)
    fig, _ = plot_ratio(altitudes, ratios, wavelength)
    fig.savefig('ratio_vs_altitude.png')

if __name__ == "__main__":
    main()
