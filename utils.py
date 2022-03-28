import numpy as np
import numba

@numba.jit(nopython=True)
def iteratePsi(psi, psinext, Uxnext, Uynext, dt, dx, N):
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0:
                # Western border:
                psinext[i, j] = Uxnext[i, j] * psinext[i + 1, j]
            elif i == N:
                # Eastern border:
                psinext[i, j] = np.conj(Uxnext[i - 1, j]) * psinext[i - 1, j]
            elif j == 0:
                # Southern border:
                psinext[i, j] = Uynext[i, j] * psinext[i, j + 1]
            elif j == N:
                # Northern border:
                psinext[i, j] = np.conj(Uynext[i, j - 1]) * psinext[i, j - 1]
            else:
                lapl_x = (Uxnext[i, j] * psinext[i + 1, j] - 2 * psinext[i, j] + \
                          np.conj(Uxnext[i - 1, j]) * psinext[i - 1, j]) / (dx ** 2)
                lapl_y = (Uynext[i, j] * psinext[i, j + 1] - 2 * psinext[i, j] + \
                          np.conj(Uynext[i, j - 1]) * psinext[i, j - 1]) / (dx ** 2)
                stress = (np.abs(psinext[i, j]) - 1) * psinext[i, j]

                psinext[i, j] = psi[i, j] + dt * (lapl_x + lapl_y - stress)

@numba.jit(nopython=True)
def forwardStepPsi(psi, psinext, Ux, Uy, dt, dx, N):
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0:
                # Western border:
                psinext[i, j] = Ux[i, j] * psi[i + 1, j]
            elif i == N:
                # Eastern border:
                psinext[i, j] = np.conj(Ux[i - 1, j]) * psi[i - 1, j]
            elif j == 0:
                # Southern border:
                psinext[i, j] = Uy[i, j] * psi[i, j + 1]
            elif j == N:
                # Northern border:
                psinext[i, j] = np.conj(Uy[i, j - 1]) * psi[i, j - 1]
            else:
                lapl_x = (Ux[i, j] * psi[i + 1, j] - 2 * psi[i, j] + \
                          np.conj(Ux[i - 1, j]) * psi[i - 1, j]) / (dx ** 2)
                lapl_y = (Uy[i, j] * psi[i, j + 1] - 2 * psi[i, j] + \
                          np.conj(Uy[i, j - 1]) * psi[i, j - 1]) / (dx ** 2)
                stress = (np.abs(psi[i, j]) - 1) * psi[i, j]

                psinext[i, j] = psi[i, j] + dt * (lapl_x + lapl_y - stress)

@numba.jit(nopython=True)
def residualPsi(psi, psinext, Uxnext, Uynext, dt, dx, N):
    residual = np.zeros_like(psi)
    for i in range(N + 1):
        for j in range(N + 1):
            if i == 0:
                # Western border:
                residual[i, j] = psinext[i, j] - Uxnext[i, j] * psinext[i + 1, j]
            elif i == N:
                # Eastern border:
                residual[i, j] = psinext[i, j] - np.conj(Uxnext[i - 1, j]) * psinext[i - 1, j]
            elif j == 0:
                # Southern border:
                residual[i, j] = psinext[i, j] - Uynext[i, j] * psinext[i, j + 1]
            elif j == N:
                # Northern border:
                residual[i, j] = psinext[i, j] - np.conj(Uynext[i, j - 1]) * psinext[i, j - 1]
            else:
                lapl_x = (Uxnext[i, j] * psinext[i + 1, j] - 2 * psinext[i, j] + \
                          np.conj(Uxnext[i - 1, j]) * psinext[i - 1, j]) / (dx ** 2)
                lapl_y = (Uynext[i, j] * psinext[i, j + 1] - 2 * psinext[i, j] + \
                          np.conj(Uynext[i, j - 1]) * psinext[i, j - 1]) / (dx ** 2)
                stress = (np.abs(psinext[i, j]) - 1) * psinext[i, j]
                dpsidt = (psinext[i, j] - psi[i, j]) / dt

                residual[i, j] = dpsidt - (lapl_x + lapl_y - stress)

    return residual


@numba.jit(nopython=True)
def iterateUx(Ux, Uxnext, psinext, Uynext, dt, dx, N, kappa, expHext, eta):
    for i in range(N):
        for j in range(N + 1):
            if j == N:
                # Northern border links
                Uxnext[i, j] = np.conj(expHext / (Uxnext[i, j - 1] * Uynext[i + 1, j - 1] * \
                                                  np.conj(Uynext[i, j - 1])))
            elif j == 0:
                # Southern border links
                Uxnext[i, j] = expHext / (Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * \
                                          np.conj(Uynext[i, j]))
            else:
                # Bulk update:
                L_below = Uxnext[i, j - 1] * Uynext[i + 1, j - 1] * np.conj(Uxnext[i, j]) * \
                          np.conj(Uynext[i, j - 1])
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * \
                         np.conj(Uynext[i, j])
                couplingterm = Uxnext[i, j] * np.imag(np.conj(psinext[i, j]) * Uxnext[i, j] * psinext[i + 1, j])
                linkterm = kappa ** 2 / (dx ** 2) * Uxnext[i, j] * (np.conj(L_below) * L_here - 1)

                Uxnext[i, j] = Ux[i, j] + eta * dt * (-1j * couplingterm - linkterm)

@numba.jit(nopython=True)
def forwardStepUx(Ux, Uxnext, psi, Uy, dt, dx, N, kappa, expHext, eta):
    for i in range(N):
        for j in range(N + 1):
            if j == N:
                # Northern border links
                Uxnext[i, j] = np.conj(expHext / (Ux[i, j - 1] * Uy[i + 1, j - 1] * \
                                                  np.conj(Uy[i, j - 1])))
            elif j == 0:
                # Southern border links
                Uxnext[i, j] = expHext / (Uy[i + 1, j] * np.conj(Ux[i, j + 1]) * \
                                          np.conj(Uy[i, j]))
            else:
                # Bulk update:
                L_below = Ux[i, j - 1] * Uy[i + 1, j - 1] * np.conj(Ux[i, j]) * \
                          np.conj(Uy[i, j - 1])
                L_here = Ux[i, j] * Uy[i + 1, j] * np.conj(Ux[i, j + 1]) * \
                         np.conj(Uy[i, j])
                couplingterm = Ux[i, j] * np.imag(np.conj(psi[i, j]) * Ux[i, j] * psi[i + 1, j])
                linkterm = kappa ** 2 / (dx ** 2) * Ux[i, j] * (np.conj(L_below) * L_here - 1)

                Uxnext[i, j] = Ux[i, j] + eta * dt * (-1j * couplingterm - linkterm)

@numba.jit(nopython=True)
def residualUx(Ux, Uxnext, psinext, Uynext, dt, dx, N, kappa, expHext, eta):
    residual = np.zeros_like(Uxnext)
    for i in range(N):
        for j in range(N + 1):
            if j == N:
                # Northern border links
                L_here = Uxnext[i, j - 1] * Uynext[i + 1, j - 1] * np.conj(Uxnext[i, j]) * np.conj(Uynext[i, j - 1])
                residual[i, j] = L_here - expHext
            elif j == 0:
                # Southern border links
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * np.conj(Uynext[i, j])
                residual[i, j] = L_here - expHext
            else:
                # Bulk update:
                L_below = Uxnext[i, j - 1] * Uynext[i + 1, j - 1] * np.conj(Uxnext[i, j]) * \
                          np.conj(Uynext[i, j - 1])
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * \
                         np.conj(Uynext[i, j])
                couplingterm = Uxnext[i, j] * np.imag(np.conj(psinext[i, j]) * Uxnext[i, j] * psinext[i + 1, j])
                linkterm = kappa ** 2 / (dx ** 2) * Uxnext[i, j] * (np.conj(L_below) * L_here - 1)
                dUxdt = (Uxnext[i, j] - Ux[i, j]) / dt
                residual[i, j] = dUxdt + eta * (linkterm + 1j * couplingterm)

    return residual

@numba.jit(nopython=True)
def iterateUy(Uy, Uynext, psinext, Uxnext, dt, dx, N, kappa, expHext, eta):
    for i in range(N + 1):
        for j in range(N):
            if i == 0:
                # Western border link
                Uynext[i, j] = np.conj(expHext / (Uxnext[i, j] * Uynext[i + 1, j] * \
                                                  np.conj(Uxnext[i, j + 1])))
            elif i == N:
                # Eastern border link
                Uynext[i, j] = expHext / (Uxnext[i - 1, j] * np.conj(Uxnext[i - 1, j + 1]) * \
                                          np.conj(Uynext[i - 1, j]))
            else:
                # Bulk update
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * \
                         np.conj(Uynext[i, j])
                L_left = Uxnext[i - 1, j] * Uynext[i, j] * np.conj(Uxnext[i - 1, j + 1]) * \
                         np.conj(Uynext[i - 1, j])
                couplingterm = Uynext[i, j] * np.imag(np.conj(psinext[i, j]) * Uynext[i, j] * psinext[i, j + 1])
                linkterm = kappa ** 2 / (dx ** 2) * Uynext[i, j] * (np.conj(L_here) * L_left - 1)

                Uynext[i, j] = Uy[i, j] + eta * dt * (-1j * couplingterm - linkterm)

@numba.jit(nopython=True)
def forwardStepUy(Uy, Uynext, psi, Ux, dt, dx, N, kappa, expHext, eta):
    for i in range(N + 1):
        for j in range(N):
            if i == 0:
                # Western border link
                Uynext[i, j] = np.conj(expHext / (Ux[i, j] * Uy[i + 1, j] * \
                                                  np.conj(Ux[i, j + 1])))
            elif i == N:
                # Eastern border link
                Uynext[i, j] = expHext / (Ux[i - 1, j] * np.conj(Ux[i - 1, j + 1]) * \
                                          np.conj(Uy[i - 1, j]))
            else:
                # Bulk update
                L_here = Ux[i, j] * Uy[i + 1, j] * np.conj(Ux[i, j + 1]) * \
                         np.conj(Uy[i, j])
                L_left = Ux[i - 1, j] * Uy[i, j] * np.conj(Ux[i - 1, j + 1]) * \
                         np.conj(Uy[i - 1, j])
                couplingterm = Uy[i, j] * np.imag(np.conj(psi[i, j]) * Uy[i, j] * psi[i, j + 1])
                linkterm = kappa ** 2 / (dx ** 2) * Uy[i, j] * (np.conj(L_here) * L_left - 1)

                Uynext[i, j] = Uy[i, j] + eta * dt * (-1j * couplingterm - linkterm)

@numba.jit(nopython=True)
def residualUy(Uy, Uynext, psinext, Uxnext, dt, dx, N, kappa, expHext, eta):
    residual = np.zeros_like(Uynext)
    for i in range(N + 1):
        for j in range(N):
            if i == 0:
                # Western border link
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * np.conj(Uynext[i, j])
                residual[i, j] = L_here - expHext
            elif i == N:
                # Eastern border link
                L_here = Uxnext[i - 1, j] * Uynext[i, j] * np.conj(Uxnext[i - 1, j + 1]) * np.conj(Uynext[i - 1, j])
                residual[i, j] = L_here - expHext

            else:
                L_here = Uxnext[i, j] * Uynext[i + 1, j] * np.conj(Uxnext[i, j + 1]) * \
                         np.conj(Uynext[i, j])
                L_left = Uxnext[i - 1, j] * Uynext[i, j] * np.conj(Uxnext[i - 1, j + 1]) * \
                         np.conj(Uynext[i - 1, j])

                dUydt = (Uynext[i, j] - Uy[i, j]) / dt
                couplingterm = Uynext[i, j] * np.imag(np.conj(psinext[i, j]) * Uynext[i, j] * psinext[i, j + 1])
                linkterm = kappa ** 2 / (dx ** 2) * Uynext[i, j] * (np.conj(L_here) * L_left - 1)

                residual[i, j] = dUydt + eta * (linkterm + 1j * couplingterm)

    return residual
