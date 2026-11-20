import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def calculate_analytical_fields(a, b, k0, theta_i, theta_s, phi_s, r, H0):
    # np.sinc(u) = sin(pi*u)/(pi*u); Balanis (11-38) through (11-40).
    eta0 = 120 * np.pi
    C = -1j * eta0 * a * b * k0 * H0 / (2 * np.pi)
    X = k0 * a/2 * np.sin(theta_s) * np.cos(phi_s)
    Y = k0 * b/2 * (np.sin(theta_s) * np.sin(phi_s) - np.sin(theta_i))
    sinc_terms = np.sinc(X / np.pi) * np.sinc(Y / np.pi)
    field_factor = np.exp(-1j * k0 * r) / r
    E_s_theta = C * field_factor * sinc_terms * np.cos(theta_s) * np.sin(phi_s)
    E_s_phi = C * field_factor * sinc_terms * np.cos(phi_s)
    return E_s_theta, E_s_phi

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

def plot_geometry(a, b, k_hat_i, E_hat_i, n_hat, alpha_i, beta_i):
    fig = go.Figure()
    # add plate
    X, Y = np.meshgrid([-a/2, a/2], [-b/2, b/2])
    fig.add_trace(go.Surface(
        x=X, y=Y, z=np.zeros_like(X), opacity=0.35,
        colorscale=[[0, 'gray'], [1, 'gray']], showscale=False,
        name='Plate', hovertemplate='x: %{x} m<br>y: %{y} m<extra>Plate</extra>'))
    
    # k, E, n
    #k_hat_i = np.array([0, np.sin(theta_i), -np.cos(theta_i)])
    #H_hat_i = np.array([1, 0, 0])
    #E_hat_i = np.array([0, np.cos(theta_i), np.sin(theta_i)])
    origin = -0.8 * max(a, b) * k_hat_i
    vectors = [k_hat_i, E_hat_i, n_hat, alpha_i, beta_i]
    labels = [r'$\hat{k}_i$', r'$\hat{E}_i$', r'$\hat{n}$', r'$\hat{\alpha}_i$', r'$\hat{\beta}_i$']
    #colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#8AB17D']
    colors = px.colors.qualitative.D3
    length = 0.2 * max(a, b)
    idx = 0
    for vector, label, color in zip(vectors, labels, colors):
        if idx > 1:
            start = np.zeros(3)
        else:
            start = origin

        end = start + length * vector
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
            mode='lines+text', text=[''], textposition='top center',
            line=dict(color=color, width=6), textfont=dict(color=color), name=label))
        fig.add_trace(go.Cone(
            x=[end[0]], y=[end[1]], z=[end[2]],
            u=[vector[0]], v=[vector[1]], w=[vector[2]],
            anchor='tip', sizemode='absolute', sizeref=0.3 * length,
            colorscale=[[0, color], [1, color]], showscale=False,
            showlegend=False, hoverinfo='skip'))
        idx += 1

    # overall layout
    extent = max(a, b)
    fig.update_layout(
        title='Incident TEₓ field and plate geometry',
        scene=dict(
            xaxis=dict(title='x [m]', range=[-extent, extent]),
            yaxis=dict(title='y [m]', range=[-extent, extent]),
            zaxis=dict(title='z [m]', range=[-0.1*extent, extent]),
            aspectmode='data', dragmode='orbit'),
        margin=dict(l=0, r=0, b=0, t=40))
    fig.show(renderer='browser', config=dict(scrollZoom=True))
    return fig

def main():
    c = 3e8
    f = 1e9
    eta0 = 120*np.pi
    wavelength = c / f
    k = 2 * np.pi * f / c

    # flat plate
    a = 5 * wavelength
    b = 5 * wavelength
    n_hat = np.array([0, 0, 1]) # x,y,z
    R_par = 1
    R_perp = -1

    # source
    theta_i = 30 * np.pi/180
    phi_i = 270 * np.pi/180
    i_hat = calc_i_hat(theta_i, phi_i)
    # local basis
    i_cross_n = np.cross(i_hat, n_hat)
    if np.isclose(np.linalg.norm(i_cross_n), 0.0, atol=1e-10):
        alpha_i = np.array([1, 0, 0])
    else:
        alpha_i = i_cross_n / np.linalg.norm(i_cross_n)
    beta_i = np.cross(i_hat, alpha_i)
    E_par = -1
    E_perp = 0
    E_0 = E_perp * alpha_i + E_par * beta_i

    # observer, N_s x 3 array
    N_s = 150
    r_mag = 100 * wavelength
    phi_s = 90 * np.pi/180
    theta_s = np.linspace(-90, 90, N_s) * np.pi/180
    r_hat_x = np.sin(theta_s)*np.cos(phi_s)
    r_hat_y = np.sin(theta_s)*np.sin(phi_s)
    r_hat_z = np.cos(theta_s)
    r = r_mag * np.stack((r_hat_x, r_hat_y, r_hat_z), axis=1)

    print(f'i_hat = {i_hat}')
    print(f'E_0 = {E_0} (x,y,z), |E_0| = {np.linalg.norm(E_0)}')
    print(f'alpha_i = {alpha_i}, beta_i = {beta_i}')
    print(f'theta_i = {theta_i*180/np.pi:.2f} deg, phi_i = {phi_i*180/np.pi:.2f} deg')
    print(f'phi_s = {phi_s*180/np.pi:.2f} deg')

    # discretizing flat plate for integration
    N_x = 200
    N_y = 200
    dx = a / N_x
    dy = b / N_y
    x_grid = np.arange(N_x) * dx + (-a/2 + dx/2)
    y_grid = np.arange(N_y) * dy + (-b/2 + dy/2)

    E_s = np.zeros((N_s, 3), dtype=complex)
    # quantities that don't depend on the observer
    E_0_dot_alpha = np.dot(E_0, alpha_i)
    E_0_dot_beta = np.dot(E_0, beta_i)
    i_dot_n = np.dot(i_hat, n_hat)
    n_cross_alpha = np.cross(n_hat, alpha_i)
    # can pull v_e and v_m outside of the integral
    v_e_1 = ((R_perp - 1) * i_dot_n * E_0_dot_alpha) * alpha_i
    v_e_2 = ((R_par + 1) * E_0_dot_beta) * n_cross_alpha
    v_e = v_e_1 - v_e_2
    v_m_1 = ((1 - R_par) * i_dot_n * E_0_dot_beta) * alpha_i
    v_m_2 = ((R_perp + 1) * E_0_dot_alpha) * n_cross_alpha
    v_m = v_m_1 - v_m_2
    # quantities that do depend on the observer
    s_hat = r / r_mag # N_s x 3
    q = i_hat - s_hat # N_s x 3
    for i_x in range(N_x):
        x_prime = x_grid[i_x]
        for i_y in range(N_y):
            y_prime = y_grid[i_y]
            r_prime = np.array([x_prime, y_prime, 0])
            phasor = np.exp(-1j * k * np.dot(q, r_prime)) # N_s
            V_2 = np.cross(s_hat, v_e) + v_m # N_s x 3
            V = np.cross(s_hat, V_2) # N_s x 3
            E_s[:, :] = E_s[:, :] + phasor[:, None] * V * dx * dy

    E_s *= 1j * k * np.exp(-1j * k * r_mag) / (4 * np.pi * r_mag)

    E_spherical = transform_cartesian_to_spherical(E_s, theta_s, phi_s)
    E_r, E_theta, E_phi = E_spherical.T

    theta_s_analytical = np.linspace(-np.pi/2, np.pi/2, 1000)
    E_theta_analytical, E_phi_analytical = calculate_analytical_fields(a, b, k, theta_i, theta_s_analytical, phi_s, r_mag, 1/eta0)

    plot_geometry(a, b, i_hat, E_0, n_hat, alpha_i, beta_i)

    fig, ax = plt.subplots()
    ax.plot(theta_s_analytical*180/np.pi, np.abs(E_theta_analytical), ':', label='analytical')
    ax.plot(theta_s*180/np.pi, np.abs(E_theta), 'x', label='numerical')
    ax.set_xlabel(r'$\theta_s$ [deg]')
    ax.set_ylabel(r'$|E_{\theta}|$')
    ax.legend()
    ax.set_title(r'$|E_{\theta}|$')

    fig2, ax2 = plt.subplots()
    ax2.plot(theta_s_analytical*180/np.pi, np.abs(E_phi_analytical), ':', label='analytical')
    ax2.plot(theta_s*180/np.pi, np.abs(E_phi), 'x', label='numerical')
    ax2.set_xlabel(r'$\theta_s$ [deg]')
    ax2.set_ylabel(r'$|E_{\phi}|$')
    ax2.legend()
    ax2.set_title(r'$|E_{\phi}|$')

    fig3, ax3 = plt.subplots()
    ax3.plot(theta_s_analytical*180/np.pi, np.zeros_like(theta_s_analytical), ':', label='analytical')
    ax3.plot(theta_s*180/np.pi, np.abs(E_r), 'x', label='numerical')
    ax3.set_xlabel(r'$\theta_s$ [deg]')
    ax3.set_ylabel(r'$|E_{r}|$')
    ax3.legend()
    ax3.set_title(r'$|E_{r}|$')

    plt.show()
    return

if __name__ == "__main__":
    main()