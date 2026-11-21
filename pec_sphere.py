import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.ticker import FuncFormatter
from scipy.special import hankel2, jv, assoc_legendre_p

def calculate_H_hat(kr, n):
    # alternative form of the spherical hankel function of the second kind
    # See balanis (10-35) for discussion
    # kr = k * r
    H = np.sqrt(np.pi * kr / 2) * hankel2(n+0.5, kr)  # balanis (11-240)
    return H

def calculate_H_hat_prime(kr, n):
    # d/d(kr) of H_hat using the product rule
    # d/dz of hankel2_n(z) = 0.5*(hankel2(n-1) - hankel2(n+1))
    term1 = (np.pi/4) * 1/(np.sqrt(np.pi * kr / 2)) * hankel2(n+0.5, kr)
    term2 = np.sqrt(np.pi * kr / 2) * 0.5 * (hankel2(n-0.5, kr) - hankel2(n+1.5, kr))
    H_prime = term1 + term2
    return H_prime

def calculate_J_hat(kr, n):
    # alternative form of the spherical Bessel function
    # see balanis (10-35) for discussion
    J = np.sqrt(np.pi * kr / 2) * jv(n+0.5, kr) 
    return J

def calculate_J_hat_prime(kr, n):
    term1 = (np.pi/4) * 1/(np.sqrt(np.pi * kr / 2)) * jv(n+0.5, kr)
    term2 = np.sqrt(np.pi * kr / 2) * 0.5 * (jv(n-0.5, kr) - jv(n+1.5, kr))
    J_prime = term1 + term2
    return J_prime

def calculate_sphere_scattering_far_field(a, f, E0_mag, r_mag, phi, theta):
    # balanis (11-242)
    # a_n defined in 11-235
    # b_n and c_n defined in 11-238
    c = 3e8
    k0 = 2 * np.pi * f / c
    N_harmonics = 20
    E_theta = 0 + 0j
    E_phi = 0 + 0j
    ka = k0 * a
    for n in range(1, N_harmonics+1):
        a_n = (1j)**(-n) * (2*n + 1) / (n*(n+1)) # balanis (11-235)
        b_n = -1*a_n * calculate_J_hat_prime(ka, n) / calculate_H_hat_prime(ka, n) # (11-238a)
        c_n = -1*a_n * calculate_J_hat(ka, n) / calculate_H_hat(ka, n) # (11-238b)

        P1, P1_prime = assoc_legendre_p(n,1,np.cos(theta),diff_n=1)

        E_theta_term1 = b_n * np.sin(theta) * P1_prime
        E_theta_term2 = c_n * P1 / np.sin(theta)
        E_theta += (1j)**n * (E_theta_term1 - E_theta_term2)

        E_phi_term1 = b_n * P1 / np.sin(theta)
        E_phi_term2 = c_n * np.sin(theta) * P1_prime
        E_phi += (1j)**n * (E_phi_term1 - E_phi_term2)

    E_theta *= 1j*E0_mag*np.exp(-1j*k0*r_mag)/(k0*r_mag)*np.cos(phi)
    E_phi *= 1j*E0_mag*np.exp(-1j*k0*r_mag)/(k0*r_mag)*np.sin(phi)
    E_r = np.zeros_like(E_theta)
    return E_r, E_theta, E_phi

def calculate_analytical_monostatic_rcs(f, a):
    # balanis (11-247)
    # a is the radius of the sphere [m], len N array
    # k and wavelength are scalars
    c = 3e8
    wavelength = c / f
    k = 2 * np.pi * f / c

    N_harmonics = 50
    ka = k * a
    series_sum = np.zeros(len(a), dtype=complex)
    for n in range(1,N_harmonics+1):
        num = (-1)**n * (2*n + 1)
        den = calculate_H_hat(ka, n) * calculate_H_hat_prime(ka, n)
        series_sum += num / den

    sigma = wavelength**2 / (4 * np.pi) * np.abs(series_sum) ** 2
    return sigma

def transform_cartesian_to_spherical(E, theta_s, phi_s):
    # follows Balanis, (II-12a)
    # E is N x 3
    # theta_s is length N
    # phi_s is a scalar

    E_spherical = np.zeros_like(E)
    for idx in range(len(theta_s)):
        theta = theta_s[idx]
        M = np.array([[np.sin(theta)*np.cos(phi_s), np.sin(theta)*np.sin(phi_s), np.cos(theta)],
                      [np.cos(theta)*np.cos(phi_s), np.cos(theta)*np.sin(phi_s), -1*np.sin(theta)],
                      [-1*np.sin(phi_s), np.cos(phi_s), 0]])
        E_spherical[idx, :] = M @ E[idx, :]
    return E_spherical

def calc_i_hat(theta_i, phi_i):
    x_i = np.sin(theta_i) * np.cos(phi_i)
    y_i = np.sin(theta_i) * np.sin(phi_i)
    z_i = np.cos(theta_i)
    i_hat = -1 * np.array([x_i, y_i, z_i])
    return i_hat

def construct_local_basis(i_hat, n_hat):
    # local basis
    i_cross_n = np.cross(i_hat, n_hat)
    if np.isclose(np.linalg.norm(i_cross_n), 0.0, atol=1e-10):
        alpha_i = np.array([1, 0, 0])
    else:
        alpha_i = i_cross_n / np.linalg.norm(i_cross_n)
    beta_i = np.cross(i_hat, alpha_i)
    return alpha_i, beta_i

def run_po_solver(a, f, E_0_mag, r_mag, phi_s, theta_s):
    # a: sphere radius, m
    # f: center frequency, Hz
    c = 3e8
    #eta0 = 120*np.pi
    wavelength = c / f
    k = 2 * np.pi * f / c

    # sphere
    n_hat = np.array([0, 0, 1]) # x,y,z
    R_par = 1
    R_perp = -1

    # source: plane wave propagating in the +z-hat direction with linear polarization in x-hat
    i_hat = np.array([0, 0, 1])
    E_0_hat = np.array([1, 0, 0])
    E_0 = E_0_mag * E_0_hat

    # observer, N_s x 3 array
    N_s = len(theta_s)
    r_hat_x = np.sin(theta_s)*np.cos(phi_s)
    r_hat_y = np.sin(theta_s)*np.sin(phi_s)
    r_hat_z = np.cos(theta_s)
    r = r_mag * np.stack((r_hat_x, r_hat_y, r_hat_z), axis=1)

    #print(f'i_hat = {i_hat}')
    #print(f'E_0 = {E_0} (x,y,z), |E_0| = {np.linalg.norm(E_0)}')
    #print(f'phi_s = {phi_s*180/np.pi:.2f} deg')

    # discretize the lit region (bottom hemisphere) of the sphere for integration
    N_theta = 100
    N_phi = 100
    theta_grid = np.linspace(np.pi/2, np.pi, N_theta)
    phi_grid = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    d_theta = theta_grid[1] - theta_grid[0]
    d_phi = phi_grid[1] - phi_grid[0]

    E_s = np.zeros((N_s, 3), dtype=complex)
    # quantities that do depend on the observer
    s_hat = r / r_mag # N_s x 3
    for i_theta in range(N_theta):
        for i_phi in range(N_phi):
            x_prime = a * np.sin(theta_grid[i_theta]) * np.cos(phi_grid[i_phi])
            y_prime = a * np.sin(theta_grid[i_theta]) * np.sin(phi_grid[i_phi])
            z_prime = a * np.cos(theta_grid[i_theta])
            r_prime = np.array([x_prime, y_prime, z_prime])

            # define n-hat and the local basis
            n_hat = r_prime / np.linalg.norm(r_prime)
            alpha_i, beta_i = construct_local_basis(i_hat, n_hat)

            # calculate the phasor term
            q = i_hat - s_hat # N_s x 3
            phasor = np.exp(-1j * k * np.dot(q, r_prime)) # N_s x 3

            # calculate the vector-valued term
            E_0_dot_alpha = np.dot(E_0, alpha_i)
            E_0_dot_beta = np.dot(E_0, beta_i)
            i_dot_n = np.dot(i_hat, n_hat)
            n_cross_alpha = np.cross(n_hat, alpha_i)
            v_e_1 = ((R_perp - 1) * i_dot_n * E_0_dot_alpha) * alpha_i
            v_e_2 = ((R_par + 1) * E_0_dot_beta) * n_cross_alpha
            v_e = v_e_1 - v_e_2
            v_m_1 = ((1 - R_par) * i_dot_n * E_0_dot_beta) * alpha_i
            v_m_2 = ((R_perp + 1) * E_0_dot_alpha) * n_cross_alpha
            v_m = v_m_1 - v_m_2
            V_2 = np.cross(s_hat, v_e) + v_m # N_s x 3
            V = np.cross(s_hat, V_2) # N_s x 3

            # what's the infinitesimal area term
            dA = a**2 * np.sin(theta_grid[i_theta]) * d_theta * d_phi

            E_s[:, :] = E_s[:, :] + phasor[:, None] * V * dA

    E_s *= 1j * k * np.exp(-1j * k * r_mag) / (4 * np.pi * r_mag)

    E_spherical = transform_cartesian_to_spherical(E_s, theta_s, phi_s)
    E_r, E_theta, E_phi = E_spherical.T
    return E_r, E_theta, E_phi

def compute_monostatic_rcs_po(f, a):
    c = 3e8
    wavelength = c / f
    N_a = len(a)
    r_mag = 100 * wavelength
    E0_mag = 1
    phi_s = 0
    theta_s = np.array([np.pi])
    rcs = np.zeros(N_a)
    for idx in range(N_a):
        E_r, E_theta, E_phi = run_po_solver(a[idx], f, E0_mag, r_mag, phi_s, theta_s)
        rcs[idx] = 4 * np.pi * r_mag**2 * np.abs(E_theta[0])**2
    return rcs

def plot_monostatic_rcs(sphere_radii, analytical_rcs, numerical_rcs, wavelength):
    # analytical trace should match balanis figure (11-29)
    fig, ax = plt.subplots()
    ax.semilogy(sphere_radii/wavelength, analytical_rcs/(np.pi*sphere_radii**2), ':', label='Analytical')
    ax.semilogy(sphere_radii/wavelength, numerical_rcs/(np.pi*sphere_radii**2), '--', label='Numerical')
    ax.set_xlabel(r'$a/\lambda_0$')
    ax.set_ylabel(r'$\sigma/\pi a^2$')
    ax.set_title('Normalized Monostatic RCS')
    ax.legend()
    ax.set_ylim(0.01, 5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val:g}')) # decimal instead of scientific notation
    return

def plot_scattered_fields(theta_s, a_lambda, E_r_a, E_theta_a, E_phi_a, E_r, E_theta, E_phi):
    fig, ax = plt.subplots()
    ax.plot(theta_s*180/np.pi, np.abs(E_theta_a), ':', label='analytical')
    ax.plot(theta_s*180/np.pi, np.abs(E_theta), '--', label='numerical')
    ax.set_xlabel(r'$\theta_s$ [deg]')
    ax.set_ylabel(r'$|E_{\theta}|$')
    ax.legend()
    ax.set_title(rf'$|E_{{\theta}}|$, $a = {a_lambda:.1f} \lambda$')

    #fig2, ax2 = plt.subplots()
    #ax2.plot(theta_s*180/np.pi, np.abs(E_phi_a), ':', label='analytical')
    #ax2.plot(theta_s*180/np.pi, np.abs(E_phi), '--', label='numerical')
    #ax2.set_xlabel(r'$\theta_s$ [deg]')
    #ax2.set_ylabel(r'$|E_{\phi}|$')
    #ax2.legend()
    #ax.set_title(rf'$|E_{{\phi}}|$, $a = {a_lambda:.1f} \lambda$')

    #fig3, ax3 = plt.subplots()
    #ax3.plot(theta_s*180/np.pi, np.abs(E_r_a), ':', label='analytical')
    #ax3.plot(theta_s*180/np.pi, np.abs(E_r), '--', label='numerical')
    #ax3.set_xlabel(r'$\theta_s$ [deg]')
    #ax3.set_ylabel(r'$|E_{r}|$')
    #ax3.legend()
    #ax.set_title(rf'$|E_{{r}}|$, $a = {a_lambda:.1f} \lambda$')
    return

def main():
    c = 3e8
    f = 1e9
    wavelength = c / f

    #r_mag = 100 * wavelength
    #a = 20 * wavelength # sphere radius
    #E0_mag = 1
    #phi_s = 0 * np.pi/180
    #theta_s = np.linspace(-180, 180, 100) * np.pi/180
    #E_r_a, E_theta_a, E_phi_a = calculate_sphere_scattering_far_field(a, f, E0_mag, r_mag, phi_s, theta_s)
    #E_r, E_theta, E_phi = run_po_solver(a, f, E0_mag, r_mag, phi_s, theta_s)
    #plot_scattered_fields(theta_s, a/wavelength, E_r_a, E_theta_a, E_phi_a, E_r, E_theta, E_phi)

    sphere_radii = np.linspace(0.02, 1.6, 100) * wavelength # m
    analytical_monostatic_rcs = calculate_analytical_monostatic_rcs(f, sphere_radii)
    po_monostatic_rcs = compute_monostatic_rcs_po(f, sphere_radii)
    plot_monostatic_rcs(sphere_radii, analytical_monostatic_rcs, po_monostatic_rcs, wavelength)

    plt.show()
    return

if __name__ == "__main__":
    main()