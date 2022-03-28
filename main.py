import numpy as np
import utils

class VortexLattice(object):
    def __init__(self, N, dx, kappa, Hext):
        self.N = N
        self.dx = dx
        self.kappa = kappa
        self.Hext = Hext

        self._initializeGrids()

    def _initializeGrids(self, scale=0.01):
        self.Ux = np.ones((self.N, self.N + 1), dtype=np.complex128)
        self.Uy = np.ones((self.N + 1, self.N), dtype=np.complex128)
        self.psi = np.random.normal(loc=.95, scale=scale, size=((self.N + 1, self.N + 1))) * np.exp(
            1j * 2 * np.pi * np.random.random(size=(self.N + 1, self.N + 1)) * 1e-7)

    @property
    def expHext(self):
        return np.exp(-1j * self.dx * self.dx * self.Hext)

    def _singlePicardIteration(self, dt, expHext, Uxnext, Uynext, psinext, eta):
        utils.iterateUx(self.Ux, Uxnext, psinext, Uynext, dt, self.dx, self.N, self.kappa, expHext, eta)
        utils.iterateUy(self.Uy, Uynext, psinext, Uxnext, dt, self.dx, self.N, self.kappa, expHext, eta)
        utils.iteratePsi(self.psi, psinext, Uxnext, Uynext, dt, self.dx, self.N)

    def _singlePicardStep(self, dt, expHext, eta, maxres=1e-5, meanres=1e-6, Nitr=None, maxitr=200, verbose=False,
                          errorval=1.5):
        Uxnext = np.copy(self.Ux)
        Uynext = np.copy(self.Uy)
        psinext = np.copy(self.psi)

        if Nitr is None:
            _maxres = 1e99
            _meanres = 1e99

            itr = 0
            while _maxres > maxres or _meanres > meanres:
                self._singlePicardIteration(dt, expHext, Uxnext, Uynext, psinext, eta)
                resUx, resUy, resPsi = self._residuals(dt, expHext, Uxnext, Uynext, psinext, eta)
                _maxres, _meanres = self._getMaxAndMean([resUx, resUy, resPsi])
                itr += 1

                if itr > maxitr:
                    raise RuntimeError("Max amount of iterations exceed!")

                if np.sum(np.abs(psinext) > errorval):
                    raise RuntimeError("Simulation is unstable. Psi is growing uncontrollably!")

            if verbose:
                print("Finished Picard iteration in", itr, "steps.")
        else:
            for itr in range(Nitr):
                self._singlePicardIteration(dt, expHext, Uxnext, Uynext, psinext, eta)

        resUx, resUy, resPsi = self._residuals(dt, expHext, Uxnext, Uynext, psinext, eta)

        self.psi = np.copy(psinext)
        self.Ux = np.copy(Uxnext) / np.abs(Uxnext)
        self.Uy = np.copy(Uynext) / np.abs(Uynext)

        return resUx, resUy, resPsi

    def _singleAndersonStep(self, dt, expHext, eta, alpha=.5, maxres=1e-5, meanres=1e-6, maxitr=200, verbose=False,
                          errorval=1.5):
        Uxnext = np.copy(self.Ux)
        Uynext = np.copy(self.Uy)
        psinext = np.copy(self.psi)

        _maxres = 1e99
        _meanres = 1e99

        itr = 0
        while _maxres > maxres or _meanres > meanres:
            Uxnextprev = np.copy(Uxnext)
            Uynextprev = np.copy(Uynext)
            psinextprev = np.copy(psinext)

            self._singlePicardIteration(dt, expHext, Uxnext, Uynext, psinext, eta)
            Uxnext = alpha*Uxnext + (1-alpha)*Uxnextprev
            Uynext = alpha*Uynext + (1-alpha)*Uynextprev
            psinext = alpha*psinext + (1-alpha)*psinextprev

            resUx, resUy, resPsi = self._residuals(dt, expHext, Uxnext, Uynext, psinext, eta)
            _maxres, _meanres = self._getMaxAndMean([resUx, resUy, resPsi])
            itr += 1

            if itr > maxitr:
                raise RuntimeError("Max amount of iterations exceed!")

            if np.sum(np.abs(psinext) > errorval):
                raise RuntimeError("Simulation is unstable. Psi is growing uncontrollably!")

        if verbose:
            print("Finished Anderson iteration in", itr, "steps.")

        resUx, resUy, resPsi = self._residuals(dt, expHext, Uxnext, Uynext, psinext, eta)

        self.psi = np.copy(psinext)
        self.Ux = np.copy(Uxnext)
        self.Uy = np.copy(Uynext)

        return resUx, resUy, resPsi

    def _singleForwardStep(self, dt, expHext):
        Uxnext = np.copy(self.Ux)
        Uynext = np.copy(self.Uy)
        psinext = np.copy(self.psi)

        utils.forwardStepPsi(self.psi, psinext, self.Ux, self.Uy, dt, self.dx, self.N)
        utils.forwardStepUx(self.Ux, Uxnext, self.psi, self.Uy, dt, self.dx, self.N, self.kappa, expHext)
        utils.forwardStepUy(self.Uy, Uynext, self.psi, self.Ux, dt, self.dx, self.N, self.kappa, expHext)

        self.psi = np.copy(psinext)
        self.Ux = np.copy(Uxnext)
        self.Uy = np.copy(Uynext)

    def _residuals(self, dt, expHext, Uxnext, Uynext, psinext, eta):
        resUx = utils.residualUx(self.Ux, Uxnext, psinext, Uynext, dt, self.dx, self.N, self.kappa, expHext, eta)
        resUy = utils.residualUy(self.Uy, Uynext, psinext, Uxnext, dt, self.dx, self.N, self.kappa, expHext, eta)
        resPsi = utils.residualPsi(self.psi, psinext, Uxnext, Uynext, dt, self.dx, self.N)

        return resUx, resUy, resPsi

    def _getMaxAndMean(self, reslist):
        maxreslist = []
        meanreslist = []
        for res in reslist:
            maxreslist.append(np.max(res))
            meanreslist.append(np.mean(res))
        return np.max(maxreslist), np.mean(meanreslist)

    def equilibriate(self, dt, eta, Nsteps=30, Nitr=30, verbose=False):
        expHext = self.expHext
        if verbose:
            print("Starting equilibriation...")
        for equibstep in range(Nsteps):
            resUx, resUy, resPsi = self._singlePicardStep(dt, expHext, eta, Nitr=Nitr)

        maxres, meanres = self._getMaxAndMean([resUx, resUy, resPsi])

        if verbose:
            print("Finished equilibriation with max residual",
                  maxres,
                  "and mean residual",
                  meanres)

        return maxres, meanres

    def _enforcenorm(self):
        self.Ux = self.Ux / np.abs(self.Ux)
        self.Uy = self.Uy / np.abs(self.Uy)

    def step(self, dt, eta, maxres=1e-5, meanres=1e-6, alpha=.5, verbose=False, enforcenorm=False, method='picard'):
        if method=='picard':
            expHext = self.expHext
            resUx, resUy, resPsi = self._singlePicardStep(dt, expHext, eta, maxres=maxres, meanres=meanres, verbose=verbose)
            if verbose:
                maxres, meanres = self._getMaxAndMean([resUx, resUy, resPsi])
                print("Performed single step to maxres", maxres, "and meanres", meanres)
        elif method=='forward':
            expHext = self.expHext
            self._singleForwardStep(dt, expHext)
        elif method=='anderson':
            expHext = self.expHext
            self._singleAndersonStep(dt, expHext, eta, alpha=alpha, maxres=maxres, meanres=meanres, verbose=verbose)
        else:
            raise ValueError("Unknown stepping method", method)

        if enforcenorm:
            self._enforcenorm()

    @property
    def amountOfVortices(self):
        totalVortices = 0
        phasemap = np.angle(self.psi) + np.pi
        i = 1
        j = 0

        while i < self.N:
            phaseDiff = phasemap[i, j] - phasemap[i - 1, j]
            if phaseDiff < 0:
                totalVortices += 1
            i += 1

        while j < self.N:
            phaseDiff = phasemap[i, j] - phasemap[i, j - 1]

            if phaseDiff < 0:
                totalVortices += 1
            j += 1

        i -= 1
        while i > 0:
            phaseDiff = phasemap[i, j] - phasemap[i + 1, j]
            if phaseDiff < 0:
                totalVortices += 1
            i -= 1

        j -= 1
        while j > 0:
            phaseDiff = phasemap[i, j] - phasemap[i, j + 1]
            if phaseDiff < 0:
                totalVortices += 1

            j -= 1

        return totalVortices

    @property
    def vortexLatticeConstantTheoretical(self):
        B = self.Hext - (1.0 - self.Hext) / ((2 * self.kappa ** 2 - 1) * 1.16)  # (5.40) in Tinkham
        vort_lat_const = 1.075 * np.sqrt(2 * np.pi / B)
        return vort_lat_const

    @property
    def vortexLatticeConstantMeasured(self):
        amountOfVortices = self.amountOfVortices
        area = self.dx * self.N * self.dx * self.N
        unitCellArea = area / amountOfVortices
        vort_lat_const = np.sqrt(2 * unitCellArea / np.sqrt(3))
        return vort_lat_const

    def addDefectLattice(self, NLatticePoints=50, useMeasuredLatticeConstant=False):
        if useMeasuredLatticeConstant:
            latticeConstant = self.vortexLatticeConstantMeasured
        else:
            latticeConstant = self.vortexLatticeConstantTheoretical
        lattice = self._generateLattice(latticeConstant, NLatticePoints)
        for point in lattice:
                indx, indy = point
                self.psi[indx, indy] = 0.0

    def setToDefectLattice(self, NLatticePoints=50, useMeasuredLatticeConstant=False):
        if useMeasuredLatticeConstant:
            latticeConstant = self.vortexLatticeConstantMeasured
        else:
            latticeConstant = self.vortexLatticeConstantTheoretical

        lattice = self._generateLattice(latticeConstant, NLatticePoints)
        for i in range(self.N):
            for j in range(self.N):
                l, k = -1, -1
                closestDistance = 1e99
                for point in lattice:
                    x, y = point
                    distance = (x-i)**2 + (y-j)**2
                    if distance < closestDistance:
                        l, k=point
                        closestDistance = distance
                # Now l,k is the closest lattice point
                r = np.sqrt(closestDistance)*self.dx
                self.psi[i,j] = np.tanh(r) * np.exp(1j*np.arctan2(k-j, l-i))

    def _generateLattice(self, latticeConstant, NLatticePoints):
        latticePoints = []
        for x in range(NLatticePoints):
            for y in range(NLatticePoints):
                posx = latticeConstant * x - .5 * latticeConstant * y
                posy = latticeConstant * np.sqrt(3) * .5 * y
                if (posx > 0.0) and (posx < self.N * self.dx):
                    if (posy > 0.0) and (posy < self.N * self.dx):
                        indx = int(round(posx / self.dx))
                        indy = int(round(posy / self.dx))
                        latticePoints.append([indx,indy])
        return latticePoints