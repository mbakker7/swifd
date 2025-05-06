import numpy as np
import pandas as pd

class SwiModel:
    
    def __init__(self, nlay, ncol, delx, xleftc=0):
        self.nlay = nlay
        self.ncol = ncol
        self.delx = delx
        self.ncell = self.nlay * self.ncol
        self.xc = np.arange(0, self.ncol * delx, delx) + xleftc
        self.Qf = []
        self.Qs = []
        self.ghbf = []
        self.ghbs = []
        self.drainf = []
        self.drains = []
        self.fixedf = []
        self.fixeds = []
        
    def tdis(self, nstep, delt, hfini, hsini):
        self.nstep = nstep
        self.delt = delt
        self.hfini = hfini
        self.hsini = hsini
    
    # def aquifer(self, k, S, Se, zb, zt, rhof, rhos):
    #     self.k = k
    #     self.S = S
    #     self.Se = Se
    #     self.zb = zb
    #     self.zt = zt
    #     if self.k.ndim == 1:
    #         self.k = self.k[:, np.newaxis]
    #     if self.S.ndim == 1:
    #         self.S = self.S[:, np.newaxis]
    #     if self.Se.ndim == 1:
    #         self.Se = self.Se[:, np.newaxis]
    #     if self.zb.ndim == 1:
    #         self.zb = self.zb[:, np.newaxis]
    #     if self.zt.ndim == 1:
    #         self.zt = self.zt[:, np.newaxis]
    #     self.H = self.zt - self.zb
    #     self.rhof = rhof
    #     self.rhos = rhos
    #     self.alphaf = self.rhof / (self.rhos - self.rhof)
    #     self.alphas = self.rhos / (self.rhos - self.rhof)

    def aquifer(self, k, S, Se, zb, zt, rhof, rhos):
        self.k = self.set_array(k, (self.nlay, 1))
        self.S = self.set_array(S, (self.nlay, 1))
        self.Se = self.set_array(Se, (self.nlay, 1))
        self.zb = self.set_array(zb, (self.nlay, self.ncol))
        self.zt = self.set_array(zt, (self.nlay, self.ncol))
        self.H = self.zt - self.zb
        self.rhof = rhof
        self.rhos = rhos
        self.alphaf = self.rhof / (self.rhos - self.rhof)
        self.alphas = self.rhos / (self.rhos - self.rhof)

    def set_array(self, var, shape):
        if np.isscalar(var):
            var = var * np.ones(shape)
        if isinstance(var, list):
            var = np.reshape(np.array(var), shape)
        else:
            if var.ndim == 1:
                if shape[1] == 1:
                    var = np.reshape(var, shape)
                else:
                    var = var[:, np.newaxis] * np.ones(shape)
            else:
                var = np.reshape(var, shape)
        # check shape is correct
        return var
        
    def set_source(self, Qf=[], Qs=[]):
        self.Qf = Qf
        self.Qs = Qs
        
    def set_ghb(self, ghbf=[], ghbs=[]):
        self.ghbf = ghbf
        self.ghbs = ghbs

    def set_drain(self, drainf=[], drains=[]):
        self.drainf = drainf
        self.drains = drains
        
    def set_fixed(self, fixedf=[], fixeds=[]):
        self.fixedf = fixedf
        self.fixeds = fixeds
        
    def cond_storage_fresh(self, hf, hfold, hs, hsold):
        hf = np.reshape(hf, (self.nlay, self.ncol))
        hfold = np.reshape(hfold, (self.nlay, self.ncol)) 
        hs = np.reshape(hs, (self.nlay, self.ncol))
        hsold = np.reshape(hsold, (self.nlay, self.ncol))
        zetaold = self.alphas * hsold - self.alphaf * hfold
        zeta = self.alphas * hs - self.alphaf * hf
        zetaold = np.maximum(zetaold, self.zb)
        zeta = np.maximum(zeta, self.zb)
        topold = np.minimum(hfold, self.zt)
        botold = np.maximum(zetaold, self.zb)
        bfold = np.maximum(topold - botold, 0)
        top = np.minimum(hf, self.zt)
        bot = np.maximum(zeta, self.zb)
        bf = np.maximum(top - bot, 0) # thickness cannot be negative
        storage1 = self.S * self.delx * (bf - bfold) / self.delt
        storage2 = self.Se * bf * self.delx * (hf - hfold) / self.delt
        # do upstream weighing after computing storage
        bf = np.where(hf[:, :-1] >= hf[:, 1:], bf[:, :-1], bf[:, 1:]) # upstream weighing
        bf = np.maximum(1e-3, bf) # make sure at least 1 mm, not needed I think
        C = np.zeros((self.nlay, self.ncol))
        C[:, :-1] = self.k * bf / self.delx
        if self.nlay > 1:
            c = 0.5 * self.H[:-1] / self.k[:-1] + 0.5 * self.H[1:] / self.k[1:]
            D = self.delx / c
        else:
            D = None
        return C, D, storage1 + storage2
        
    def jac(self, x, hold, *args, fun=None):
        dp = 1e-6
        ntot = len(x)
        d = dp * np.eye(ntot)
        rv = np.zeros((ntot, ntot))
        funx = fun(x, hold, *args)
        for n in range(ntot):
            rv[:, n] = (fun(x + d[n], hold, *args) - 
                        fun(x - d[n], hold, *args)) / (2 * dp)
        return rv
        
    def step_fresh(self, hf, hfold, hs, hsold):
        C, D, storage = self.cond_storage_fresh(hf, hfold, hs, hsold)
        #
        A = np.diag(C.ravel()[:-1], 1) + np.diag(C.ravel()[:-1], -1)
        if self.nlay > 1:
            A += np.diag(D.ravel(), self.ncol) + np.diag(D.ravel(), -self.ncol) 
        A -= np.diag(A.sum(1))
        rhs = storage.ravel()
        for ilay, icol, Q in self.Qf:
            index = ilay * self.ncol + icol
            rhs[index] -= Q
        for ilay, icol, hstar, Cstar in self.ghbf:
            index = ilay * self.ncol + icol
            A[index, index] -= Cstar
            rhs[index] -= Cstar * hstar
        for ilay, icol, hstar, Cstar in self.drainf:
            index = ilay * self.ncol + icol
            if hf[index] > hstar:
                A[index, index] -= Cstar
                rhs[index] -= Cstar * hstar
        for ilay, icol, hfixed in self.fixedf:
            index = ilay * self.ncol + icol
            A[index] = 0
            A[index, index] = 1.0
            rhs[index] = hfixed
        sol = A @ hf - rhs
        return sol

    def budget_fresh_step(self, hf, hfold, hs=None, hsold=None):
        Qsource = np.zeros(self.nlay)
        Qfixed = np.zeros(self.nlay)
        Qghb = np.zeros(self.nlay)
        Qdrain = np.zeros(self.nlay)
        Qtop = np.zeros(self.nlay)
        Qbot = np.zeros(self.nlay)
        C, D, storage = self.cond_storage_fresh(hf, hfold, hs, hsold)
        # vertical flow
        if self.nlay > 1:
            Qml = D * (hf[1:] - hf[:-1]) * self.delt
        else:
            Qml = np.zeros((1, self.ncol))
        # horizontal flow
        Qx = C[:, :-1] * (hf[:, :-1] - hf[:, 1:]) * self.delt
        #
        Qsource_array = np.zeros((self.nlay, self.ncol))
        for ilay, jcol, Q in self.Qf:
            Qsource_array[ilay, jcol] = Q * self.delt # needed to correct for fixed head cells
        for ilay, jcol, hfixed in self.fixedf:
            if jcol < self.ncol - 1:
                Qfixed[ilay] += Qx[ilay, jcol]
            if jcol > 0:
                Qfixed[ilay] -= Qx[ilay, jcol - 1]
            if ilay < self.nlay - 1:
                Qfixed[ilay + 1] -= Qml[ilay, jcol]
                Qml[ilay, jcol] = 0
            if ilay > 0:
                Qfixed[ilay - 1, jcol] += Qm[ilay - 1, jcol]
                Qm[ilay - 1, jcol] = 0
            Qsource_array[ilay, jcol] = 0 # no source on constant head cells   
        Qsource = np.sum(Qsource_array, axis=1)
        Qbot[:-1] = np.sum(Qml, axis=1)
        Qtop[1:] = np.sum(-Qml, axis=1)
        #
        for ilay, jcol, hstar, Cstar in self.ghbf:
            Qghb[ilay] += Cstar * (hstar - hf[ilay, jcol]) * self.delt
        for ilay, jcol, hstar, Cstar in self.drainf:
            if hf[ilay, jcol] > hstar:
                Qdrain[ilay] += Cstar * (hstar - hf[ilay, jcol]) * self.delt
        storage_increase = storage * self.delt
        storage_increase = np.reshape(storage_increase, (self.nlay, self.ncol))
        storage_increase = np.sum(storage_increase, axis=1)
        in_min_out = Qsource + Qfixed + Qghb + Qdrain + Qtop + Qbot
        balance = in_min_out - storage_increase
        rv = pd.DataFrame(np.array((Qsource, Qfixed, Qghb, Qdrain, Qtop, Qbot, storage_increase, in_min_out, balance)),
             index=['Source', 'Fixed', 'GHB', 'Drain', 'Qtop', 'Qbot', 'storage_increase', 'in_min_out', 'balance'],
             columns=['layer ' + str(ilay) for ilay in np.arange(self.nlay)])
        rv['total'] = rv.sum(axis=1)
        return rv

    def budget_fresh(self, hfsol, hssol=None):
        if hssol is None: # freshonly
            hssol = self.hsini * np.ones((self.nstep + 1, self.nlay, self.ncol))
        rv = np.zeros((self.nlay + 1, self.nstep, 9))
        for istep in range(self.nstep):
            budget = self.budget_fresh_step(hfsol[istep + 1], hfsol[istep], hssol[istep + 1], hssol[istep])
            for ilay in range(self.nlay + 1):
                rv[ilay, istep] = budget.to_numpy()[:, ilay]
        rvdic = {}
        for ilay, key in enumerate(budget.columns): # last one is total
            rvdic[key] = (pd.DataFrame(data=rv[ilay], columns=budget.index))
        return rvdic
                            
    def cond_storage_salt(self, hs, hsold, hf, hfold):
        hf = np.reshape(hf, (self.nlay, self.ncol))
        hfold = np.reshape(hfold, (self.nlay, self.ncol)) 
        hs = np.reshape(hs, (self.nlay, self.ncol))
        hsold = np.reshape(hsold, (self.nlay, self.ncol))
        zetaold = self.alphas * hsold - self.alphaf * hfold
        zeta = self.alphas * hs - self.alphaf * hf
        topold = np.minimum(zetaold, self.zt)
        bsold = np.maximum(topold - self.zb, 0)
        top = np.minimum(zeta, self.zt)
        bs = np.maximum(top - self.zb, 0) # bs cannot be negative
        storage1 = self.S * self.delx * (bs - bsold) / self.delt
        storage2 = self.Se * bs * self.delx * (hs - hsold) / self.delt
        bs = np.where(hs[:, :-1] >= hs[:, 1:], bs[:, :-1], bs[:, 1:]) # upstream weighing
        bs = np.maximum(1e-3, bs) # make sure at least 1 mm
        C = np.zeros((self.nlay, self.ncol))
        C[:, :-1] = self.k * bs / self.delx
        c = 0.5 * self.H[:-1] / self.k[:-1] + 0.5 * self.H[1:] / self.k[1:]
        D = self.delx / c
        return C, D, storage1 + storage2

    def step_salt(self, hs, hsold, hf, hfold):
        C, D, storage = self.cond_storage_salt(hs, hsold, hf, hfold)
        A = np.diag(C.ravel()[:-1], 1) + np.diag(C.ravel()[:-1], -1)
        if self.nlay > 1:
            A += np.diag(D.ravel(), self.ncol) + np.diag(D.ravel(), -self.ncol) 
        A -= np.diag(A.sum(1))
        rhs = storage.ravel()
        for ilay, icol, Q in self.Qs:
            index = ilay * self.ncol + icol
            rhs[index] -= Q
        for ilay, icol, hstar, Cstar in self.ghbs:
            index = ilay * self.ncol + icol
            A[index, index] -= Cstar
            rhs[index] -= Cstar * hstar
        for ilay, icol, hstar, Cstar in self.drains:
            index = ilay * self.ncol + icol
            if hf[index] > hstar:
                A[index, index] -= Cstar
                rhs[index] -= Cstar * hstar
        for ilay, icol, hfixed in self.fixeds:
            index = ilay * self.ncol + icol
            A[index] = 0
            A[index, index] = 1.0
            rhs[index] = hfixed
        return A @ hs - rhs # still doing saltwater here, but hf and zeta should do the same

    def budget_salt_step(self, hs, hsold, hf, hfold):
        Qsource = np.zeros(self.nlay)
        Qfixed = np.zeros(self.nlay)
        Qghb = np.zeros(self.nlay)
        Qdrain = np.zeros(self.nlay)
        Qtop = np.zeros(self.nlay)
        Qbot = np.zeros(self.nlay)
        C, D, storage = self.cond_storage_salt(hs, hsold, hf, hfold)
        # vertical flow
        if self.nlay > 1:
            Qml = D * (hs[1:] - hs[:-1]) * self.delt
        else:
            Qml = np.zeros((1, self.ncol))
        # horizontal flow
        Qx = C[:, :-1] * (hs[:, :-1] - hs[:, 1:]) * self.delt
        #
        Qsource_array = np.zeros((self.nlay, self.ncol))
        for ilay, jcol, Q in self.Qs:
            Qsource_array[ilay, jcol] = Q * self.delt # needed to correct for fixed head cells
        #
        for ilay, jcol, hfixed in self.fixeds:
            if jcol < self.ncol - 1:
                Qfixed[ilay] += Qx[ilay, jcol]
            if jcol > 0:
                Qfixed[ilay] -= Qx[ilay, jcol - 1]
            if ilay < self.nlay - 1:
                Qfixed[ilay + 1] -= Qml[ilay, jcol]
                Qml[ilay, jcol] = 0
            if ilay > 0:
                Qfixed[ilay - 1, jcol] += Qm[ilay - 1, jcol]
                Qm[ilay - 1, jcol] = 0
            Qsource_array[ilay, jcol] = 0 # no source on constant head cells   
        Qsource = np.sum(Qsource_array, axis=1)
        Qbot[:-1] = np.sum(Qml, axis=1)
        Qtop[1:] = np.sum(-Qml, axis=1)
        for ilay, jcol, hstar, Cstar in self.ghbs:
            Qghb[ilay] += Cstar * (hstar - hs[ilay, jcol]) * self.delt
        for ilay, jcol, hstar, Cstar in self.drains:
            if hf[ilay, jcol] > hstar:
                Qdrain[ilay] += Cstar * (hstar - hs[ilay, jcol]) * self.delt
        storage_increase = storage * self.delt
        storage_increase = np.reshape(storage_increase, (self.nlay, self.ncol))
        storage_increase = np.sum(storage_increase, axis=1)
        in_min_out = Qsource + Qfixed + Qghb + Qdrain + Qtop + Qbot
        balance = in_min_out - storage_increase
        rv = pd.DataFrame(np.array((Qsource, Qfixed, Qghb, Qdrain, Qtop, Qbot, storage_increase, in_min_out, balance)),
             index=['Source', 'Fixed', 'GHB', 'Drain', 'Qtop', 'Qbot', 'storage_increase', 'in_min_out', 'balance'],
             columns=['layer ' + str(ilay) for ilay in np.arange(self.nlay)])
        rv['total'] = rv.sum(axis=1)
        return rv

    def budget_salt(self, hfsol, hssol):
        rv = np.zeros((self.nlay + 1, self.nstep, 9))
        for istep in range(self.nstep):
            budget = self.budget_salt_step(hssol[istep + 1], hssol[istep], hfsol[istep + 1], hfsol[istep])
            for ilay in range(self.nlay + 1):
                rv[ilay, istep] = budget.to_numpy()[:, ilay]
        rvdic = {}
        for ilay, key in enumerate(budget.columns): # last one is total
            rvdic[key] = (pd.DataFrame(data=rv[ilay], columns=budget.index))
        return rvdic

    def step_fresh_salt(self, solnew, solold):
        # solnew is vector with fresh heads and then zeta
        hf = solnew[:self.ncell]
        hs = solnew[self.ncell:]
        hfold = solold[:self.ncell]
        hsold = solold[self.ncell:]
        fresh_sol = self.step_fresh(hf, hfold, hs, hsold)
        salt_sol = self.step_salt(hs, hsold, hf, hfold)
        return np.hstack((fresh_sol, salt_sol))

    def simulate_freshonly(self, maxiter=100, silent=False):
        htol = 1e-6 # absolute convergence criterium for heads
        sol = np.zeros((self.nstep + 1, self.nlay * self.ncol))
        sol[0] = self.hfini.flatten()
        hs = self.hsini.flatten()
        for istep in range(self.nstep):
            solnew = sol[istep] + 0.1 # alter a bit to start?
            for jiter in range(maxiter):
                sol[istep + 1] = solnew
                R = self.step_fresh(solnew, sol[istep], hs, hs)
                J = self.jac(solnew, sol[istep], hs, hs, fun=self.step_fresh)
                solnew = np.linalg.solve(J, -R + J @ solnew)
                if np.max(np.abs(sol[istep + 1] - solnew)) < htol:
                    if not silent:
                        print(f'iterations: {jiter + 1}')
                    sol[istep + 1] = solnew
                    break
                if jiter == maxiter - 1:
                    print(f'zero based time step: {istep}')
                    print(f'Error: convergence not reached after maxiter={maxiter} iterations')
                    sol[istep + 1] = solnew
                    break
        hfsol = sol
        hfsol = np.reshape(sol, (self.nstep + 1, self.nlay, self.ncol))
        zetasol = self.alphas * self.hsini - self.alphaf * hfsol
        zetasol = np.maximum(zetasol, self.zb)
        zetasol = np.minimum(zetasol, self.zt)
        return hfsol, zetasol

    def simulate(self, maxiter=100, silent=False, test=False):
        htol = 1e-6 # absolute convergence criterium for heads
        sol = np.zeros((self.nstep + 1, 2 * self.ncell))
        sol[0, :self.ncell] = self.hfini.flatten()
        sol[0, self.ncell:] = self.hsini.flatten()
        for istep in range(self.nstep):
            solnew = sol[istep]
            #print(solnew)
            for jiter in range(maxiter):
                #print('istep, jiter ', istep, jiter)
                sol[istep + 1] = solnew
                R = self.step_fresh_salt(solnew, sol[istep])
                J = self.jac(solnew, sol[istep], fun=self.step_fresh_salt)
                if test:
                    np.save('Rmat.npy', R)
                    np.save('Jmat.npy', J)
                solnew = np.linalg.solve(J, -R + J @ solnew)
                if np.max(np.abs(sol[istep + 1] - solnew)) < htol:
                    if not silent:
                        print(f'iterations: {jiter + 1}')
                    sol[istep + 1] = solnew
                    break
                if jiter == maxiter - 1:
                    print(f'zero based time step: {istep}')
                    print(f'Error: convergence not reached after maxiter={maxiter} iterations')
                    sol[istep + 1] = solnew
                    break
        hfsol = sol[:, :self.ncell]
        hssol = sol[:, self.ncell:]
        hfsol = np.reshape(hfsol, (self.nstep + 1, self.nlay, self.ncol))
        hssol = np.reshape(hssol, (self.nstep + 1, self.nlay, self.ncol))
        zetasol = self.alphas * hssol - self.alphaf * hfsol
        zetasol = np.maximum(zetasol, self.zb) # zeta is solution variable, so need to adjust
        zetasol = np.minimum(zetasol, self.zt)
        return hfsol, hssol, zetasol