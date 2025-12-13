import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scqubits

plt.close("all")

# scqubit


def get_eigen_sc(flux, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
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
    e1, e2, e3 = get_eigen_sc(flux=phi_ext)
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


def get_eigen_qu(flux_val, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
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
    e1, e2, e3 = get_eigen_qu(f)
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


def get_eigen_np(flux_val, EJ=8.9, EC=2.5, EL=0.5, cutoff=110):
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
    e1, e2, e3 = get_eigen_np(f)
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


# Comparaison de convergence

cutoff_min = 6
cutoff_max = 110
CUT = np.arange(cutoff_min,cutoff_max)
true_value = get_eigen_sc(flux = 0,cutoff = 30)
Res = np.zeros(
    (3, 3, cutoff_max-cutoff_min)
)  # indice i indique le niveau d'énergie, indice j la méthode utilisee et indice k le cutoff
for cut in CUT:
    E_sc = get_eigen_sc(flux = 0, cutoff=cut) - true_value
    E_qu = get_eigen_qu(flux_val=0, cutoff=cut) - true_value
    E_np = get_eigen_np(flux_val=0, cutoff=cut) - true_value
    Res[:, :, cut-cutoff_min] = np.array([E_sc, E_qu, E_np]).T

plt.figure(4)
plt.plot(CUT,Res[0][0][:],label = r'Methode sc')
plt.plot(CUT,Res[0][1][:],label = r'Methode qutip')
plt.plot(CUT,Res[0][2][:],label = r'Methode numpy')
plt.xlabel(r"cutoff")
plt.ylabel(r"$E^{(1)}-E^{(0)}$ [GHz]")
plt.title(r'Comparaison de la convergence pour $E^{(1)} - E^{(0)}$')
plt.legend()
plt.show()

plt.figure(5)
plt.plot(CUT,Res[1][0][:],label = r'Methode sc')
plt.plot(CUT,Res[1][1][:],label = r'Methode qutip')
plt.plot(CUT,Res[1][2][:],label = r'Methode numpy')
plt.xlabel(r"cutoff")
plt.ylabel(r"$E^{(2)}-E^{(0)}$ [GHz]")
plt.title(r'Comparaison de la convergence pour $E^{(2)} - E^{(0)}$')
plt.legend()
plt.show()

plt.figure(6)
plt.plot(CUT,Res[2][0][:],label = r'Methode sc')
plt.plot(CUT,Res[2][1][:],label = r'Methode qutip')
plt.plot(CUT,Res[2][2][:],label = r'Methode numpy')
plt.xlabel(r"cutoff")
plt.ylabel(r"$E^{(3)}-E^{(0)}$ [GHz]")
plt.title(r'Comparaison de la convergence pour $E^{(3)} - E^{(0)}$')
plt.legend()
plt.show()