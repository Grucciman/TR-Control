import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scqubits

plt.close("all")

# scqubit


def get_eigen(flux, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
    fluxonium = scqubits.Fluxonium(
        EJ=EJ, EC=EC, EL=EL, flux=flux / (2 * np.pi), cutoff=cutoff
    )
    H = fluxonium.hamiltonian()
    vals, vecs = fluxonium.eigensys()
    vals = vals - vals[0]
    return vals[1:4]


E_1 = []
E_2 = []
E_3 = []
flux = np.linspace(0, 1, 100)
for phi_ext in flux:
    e1, e2, e3 = get_eigen(flux=phi_ext)
    E_1.append(e1), E_2.append(e2), E_3.append(e3)

plt.figure(1)
plt.plot(flux, E_1, label=r"$E^{(1)}$")
plt.plot(flux, E_2, label=r"$E^{(2)}$")
plt.plot(flux, E_3, label=r"$E^{(3)}$")
plt.xlabel("$\phi_{ext}$ [rad]")
plt.ylabel("$E^{(n)}-E^{(0)}$ [GHz]")
plt.title("Résolution avec scqubit")
plt.legend()


# qutip


def get_eigen(flux_val, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
    phi_ext = flux_val

    omega = np.sqrt(8 * EL * EC)
    phi_zpf = (8 * EC / EL) ** 0.25

    a = qt.destroy(cutoff)
    phi = (a + a.dag()) / np.sqrt(2)
    phi = phi_zpf * phi

    H = omega * (a.dag() * a + 0.5) - EJ * (phi - phi_ext).cosm()

    energies = H.eigenenergies()
    energies = energies - energies[0]
    return energies[1:4]


flux_list = np.linspace(0, 1, 100)
E_1, E_2, E_3 = [], [], []

for f in flux_list:
    e1, e2, e3 = get_eigen(f)
    E_1.append(e1)
    E_2.append(e2)
    E_3.append(e3)

plt.figure(2)
plt.plot(flux_list, E_1, label=r"$E^{(1)}$")
plt.plot(flux_list, E_2, label=r"$E^{(2)}$")
plt.plot(flux_list, E_3, label=r"$E^{(3)}$")
plt.xlabel(r"$\phi_{ext}$ [rad]")
plt.ylabel(r"$E^{(n)}-E^{(0)}$ [GHz]")
plt.title("Résolution avec Qutip")
plt.legend()


# Numpy


def get_eigen(flux_val, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
    phi_ext = flux_val

    omega = np.sqrt(8 * EL * EC)
    phi_zpf = (8 * EC / EL) ** 0.25

    a = np.zeros((cutoff, cutoff))
    for i in range(cutoff - 1):
        a[i][i + 1] = np.sqrt(i + 1)
    a_dag = a.conj().transpose()
    phi = (a + a_dag) / np.sqrt(2)
    phi = phi_zpf * phi

    eigen_values, eigen_vectors = np.linalg.eigh(phi - phi_ext * np.eye(cutoff))
    P = eigen_vectors
    cos_D = np.diag(np.cos(eigen_values))

    cos_phi = P @ cos_D @ P.transpose().conj()

    H = omega * (a_dag @ a + 0.5 * np.eye(cutoff)) - EJ * cos_phi

    eigen_values_H = np.linalg.eigvalsh(H)

    energies = eigen_values_H - eigen_values_H[0]
    return energies[1:4]


flux_list = np.linspace(0, 1, 100)
E_1, E_2, E_3 = [], [], []

for f in flux_list:
    e1, e2, e3 = get_eigen(f)
    E_1.append(e1)
    E_2.append(e2)
    E_3.append(e3)

plt.figure(3)
plt.plot(flux_list, E_1, label=r"$E^{(1)}$")
plt.plot(flux_list, E_2, label=r"$E^{(2)}$")
plt.plot(flux_list, E_3, label=r"$E^{(3)}$")
plt.xlabel(r"$\phi_{ext}$ [rad]")
plt.ylabel(r"$E^{(n)}-E^{(0)}$ [GHz]")
plt.title("Résolution avec Numpy")
plt.legend()


plt.show()
