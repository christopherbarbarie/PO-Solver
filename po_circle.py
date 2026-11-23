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

def main():
    wavelength = 10
    h = 32000 * wavelength
    a = fresnel_zone(h, wavelength)
    E = flat_disk(h, a, wavelength)
    I = 2 * np.pi * h**2 * abs(E[0]) / wavelength**2
    print(I**2, (h / wavelength)**2)

if __name__ == "__main__":
    main()
    print("hello world :O)")
