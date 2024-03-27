import itertools
import numpy as np
import time
import scipy.sparse 
from scipy.sparse import linalg as splinalg


class MarkovChain(object):
    def __init__(self, L_V = 6, L_B = 1, L_E = 4, P=None, init_dist=None, verbose=False, doubleacc=True):
        self.verbose = verbose
        
        t0 = time.time()

        if P is not None: self.P = P
        else:
            self.P =np.asarray([[1,1,0,2,0],
                                [1,1,1,1,0],
                                [0,1,1,1,0],
                                [2,0,1,0,1],
                                [1,0,1,0,1]], dtype=float)
            self.P = self.P/np.sum(self.P, axis=1)[:,None]

            
        self.N_states = self.P.shape[0]
        
        self.L_V = L_V
        self.L_B = L_B
        self.L_E = L_E
        
        N_V_configs = self.N_states**self.L_V
        N_E_configs = self.N_states**self.L_E
        
        self.eigdecomp()
        self.init_dist = self.steady if init_dist is None else init_dist
        self.load_chain_configs()
        self.load_PV_PE(self.init_dist, doubleacc=doubleacc)
        
        if init_dist is not None:
            self.steady = init_dist.copy()
            self.init_dist = init_dist.copy()
            self.p_v = self.steady
            self.p_e = np.einsum('ji,i->j', self.P_E_cond_V, self.p_v) # sum p(e|v)p(v)
        else:
            self.p_v = self.steady
            self.p_e = np.einsum('ji,i->j', self.P_E_cond_V, self.p_v) # sum p(e|v)p(v)
            
        if self.verbose: print(f"Size of P(E|V):\t{self.P_E_cond_V.shape, }\n{time.time()-t0:.2f} seconds to initialize")
        
        
    def eigdecomp(self):
        wr, vr = np.linalg.eig(self.P)
        wl, vl = np.linalg.eig(self.P.T)
        
        wr_argsort = np.argsort(np.abs(wr))[::-1] # args sorted from max to min (descending order)
        wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)

        vr_sorted = vr.T[wr_argsort]
        vl_sorted = vl.T[wl_argsort]

        steady = vl_sorted[0].real
        #print("steady sum:\t", np.sum(steady))
        steady /= np.sum(steady)   
        self.steady = steady

        wr_argsort = np.argsort(np.abs(wr))[::-1] # args sorted from max to min (descending order)
        wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)
        
        self.vr_sort = vr.T[wr_argsort]
        self.vl_sort = vl.T[wl_argsort]
        self.evals = wr[wr_argsort]

        def eigval(self, idx, mode='right'):
            return self.vr_sort[idx] if mode=='right' else self.vl_sort[idx]

        def eigvec(self, idx):
            return self.evals[idx]

        R1 = vr.T[wr_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect
        L1 = vl.T[wl_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect
        self.R1 = R1
        self.L1 = L1.real
        self.lambd_1 = wr[wr_argsort][1].real
        if len(wr)>3:

            
            self.lambd_2 = wr[wr_argsort][2].real
            self.lambd_3 = wr[wr_argsort][3].real


            R2 = vr.T[wr_argsort][2] # First sorts the columns (rows after transpose) and gets second eigenvect
            L2 = vl.T[wl_argsort][2] # First sorts the columns (rows after transpose) and gets second eigenvect

            R3 = vr.T[wr_argsort][3] # First sorts the columns (rows after transpose) and gets second eigenvect
            L3 = vl.T[wl_argsort][3] # First sorts the columns (rows after transpose) and gets second eigenvect

            R4 = vr.T[wr_argsort][4] # First sorts the columns (rows after transpose) and gets second eigenvect
            L4 = vl.T[wl_argsort][4] # First sorts the columns (rows after transpose) and gets second eigenvect

            if self.verbose: print("IS R1 REAL? SUM IMAG: ", R1.imag.sum())
            if self.verbose: print("IS L1 REAL? SUM IMAG: ", L1.imag.sum())


            self.R2 = R2
            self.L2 = L2.real

            self.R3 = R3
            self.L3 = L3.real

            self.R4 = R4
            self.L4 = L4.real
        
        return
        
        
    def load_chain_configs(self):
        self.V_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_V))
        self.E_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_E))

        if len(self.V_configs)*len(self.E_configs) > 1e9: assert 1==0

        self.VR_configs = [V[-1] for V in self.V_configs]
        self.EL_configs = [E[0] for E in self.E_configs]
        
        return
    
    def load_PV_PE(self, init_dist=None, doubleacc=False):
        P_buffer = np.linalg.matrix_power(self.P, self.L_B+1)
        
        if doubleacc:
            if self.verbose: print("Using double acc")
            P_buffer = P_buffer.astype(np.longdouble)
            init_dist = init_dist.astype(np.longdouble)
            self.P = self.P.astype(np.longdouble)
            
        self.P_V = np.asarray(init_dist)
        #self.P_E = np.asarray(init_dist @ self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs])
        #sself.P_E_free = np.asarray([np.prod([self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs] )
        # Note: P_E_free above isn't a true probability distribution, since the initial state is free
        #P_E_joint_V = np.asarray([ [ pv*pe*P_buffer[VR_configs[iv], EL_configs[ie]] for iv, pv in enumerate(P_V)] for ie, pe in enumerate(P_E_free)])
        self.P_E_cond_V = P_buffer.T #@ P_buffer
        
        self.P_E = self.P_V[None,:] @ P_buffer
        #self.P_E_cond_V = np.asarray([ [ pe*P_buffer[self.VR_configs[iv], self.EL_configs[ie]] for iv, pv in enumerate(self.P_V)] for ie, pe in enumerate(self.P_E_free)])
        
        self.P_buffer=P_buffer
        
        #print(self.P_E, np.sum(self.P_E))


        #self.P_V = np.asarray([init_dist[V[0]]*np.prod([self.P[v_prv, v_nxt] for v_prv, v_nxt in zip(V[:-1], V[1:])]) for V in self.V_configs])
        
        #self.P_E = np.asarray([init_dist[E[0]]*np.prod([self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs])
        #self.P_E_free = np.asarray([np.prod([self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs] )
        # Note: P_E_free above isn't a true probability distribution, since the initial state is free
        #P_E_joint_V = np.asarray([ [ pv*pe*P_buffer[VR_configs[iv], EL_configs[ie]] for iv, pv in enumerate(P_V)] for ie, pe in enumerate(P_E_free)])
        #old = np.asarray([ [ pe*P_buffer[self.VR_configs[iv], self.EL_configs[ie]] for iv, pv in enumerate(self.P_V)] for ie, pe in enumerate(self.P_E_free)])
        #print(old.shape, self.P_E_cond_V.shape)

        #self.P_buffer=P_buffer

        return

    
    
    
class SparseMarkovChain(object):
    def __init__(self, L_V = 6, L_B = 1, L_E = 4, P=None, init_dist=None, verbose=False, doubleacc=True, n_evals=100):
        """
        Assume that P is a sparse matrix
        """
        self.verbose = verbose
        self.P = P
        
        t0 = time.time()
            
        self.N_states = self.P.shape[0]
        
        self.L_V = L_V
        self.L_B = L_B
        self.L_E = L_E
        
        N_V_configs = self.N_states**self.L_V
        N_E_configs = self.N_states**self.L_E
        
        self.eigdecomp(n_evals=n_evals)
        self.init_dist = self.steady if init_dist is None else init_dist
        self.load_chain_configs()
        self.load_PV_PE(self.init_dist, doubleacc=doubleacc)
        
        if self.verbose: print(f"Size of P(E|V):\t{self.P_E_cond_V.shape, }\n{time.time()-t0:.2f} seconds to initialize")
        
        
    def eigdecomp(self, n_evals):
        wr, vr = splinalg.eigs(self.P, k=n_evals)
        wl, vl = splinalg.eigs(self.P.T, k=n_evals)
        
        wr_argsort = np.argsort(np.abs(wr))[::-1] # args sorted from max to min (descending order)
        wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)

        vr_sorted = vr.T[wr_argsort]
        vl_sorted = vl.T[wl_argsort]

        steady = vl_sorted[0].real
        #print("steady sum:\t", np.sum(steady))
        steady /= np.sum(steady)   
        self.steady = steady

        wr_argsort = np.argsort(np.abs(wr))[::-1] # args sorted from max to min (descending order)
        wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)

        self.lambd_1 = wr[wr_argsort][1].real
        self.lambd_2 = wr[wr_argsort][2].real
        self.lambd_3 = wr[wr_argsort][3].real

        R1 = vr.T[wr_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect
        L1 = vl.T[wl_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect
        
        R2 = vr.T[wr_argsort][2] # First sorts the columns (rows after transpose) and gets second eigenvect
        L2 = vl.T[wl_argsort][2] # First sorts the columns (rows after transpose) and gets second eigenvect

        R3 = vr.T[wr_argsort][3] # First sorts the columns (rows after transpose) and gets second eigenvect
        L3 = vl.T[wl_argsort][3] # First sorts the columns (rows after transpose) and gets second eigenvect

        R4 = vr.T[wr_argsort][4] # First sorts the columns (rows after transpose) and gets second eigenvect
        L4 = vl.T[wl_argsort][4] # First sorts the columns (rows after transpose) and gets second eigenvect


        self.vr_sort = vr.T[wr_argsort]
        self.vl_sort = vl.T[wl_argsort]
        self.evals = wr[wr_argsort]
        
        def eigval(self, idx, mode='right'):
            return self.vr_sort[idx] if mode=='right' else self.vl_sort[idx]
        
        def eigvec(self, idx):
            return self.evals[idx]
        
        self.R1 = R1
        self.L1 = L1.real
        
        self.R2 = R2
        self.L2 = L2.real
        
        self.R3 = R3
        self.L3 = L3.real
        
        self.R4 = R4
        self.L4 = L4.real
        
        return
        
        
    def load_chain_configs(self):
        self.V_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_V))
        self.E_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_E))

        if len(self.V_configs)*len(self.E_configs) > 1e9: assert 1==0

        self.VR_configs = [V[-1] for V in self.V_configs]
        self.EL_configs = [E[0] for E in self.E_configs]
        
        return
    
    def load_PV_PE(self, init_dist=None, doubleacc=False):
        P_buffer = self.P.copy()
        
        for _ in range(self.L_B):
            P_buffer = P_buffer @ self.P #np.linalg.matrix_power(self.P, self.L_B+1)
        
        if doubleacc:
            if self.verbose: print("Using double acc")
            P_buffer = P_buffer.astype(np.longdouble)
            init_dist = init_dist.astype(np.longdouble)
            self.P = self.P.astype(np.longdouble)
            
        self.P_V = np.asarray(init_dist)
        self.P_E_cond_V = P_buffer.T #@ P_buffer
        
        self.P_E = self.P_V[None,:] @ P_buffer
        
        self.P_buffer=P_buffer
        
        return    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class DisjointChain(object):
    
    def __init__(self, L_V = 6, L_B = 1, L_E = 4, P=None, init_dist=None, verbose=False):
        self.verbose = verbose
        
        t0 = time.time()
        
        if isinstance(P, np.ndarray): self.P = P
        elif P==1:
            self.P = np.asarray([[1,2,0,0,0],
                                 [1,1,0,0,0],
                                 [0,1,1,1,0],
                                 [0,0,0,2,1],
                                 [0,0,0,1,1]], dtype=float)
        elif P==2:
            self.P = np.asarray([[1,0,0,0,0],
                                [0,1,0,0,0],
                                [1,1,2,1,1],
                                [0,0,0,1,0],
                                [0,0,0,0,1]], dtype=float)
        elif P==3:
            self.P = np.asarray([[1,1,0,0,0],
                                [1,1,0,0,0],
                                [0,0,1,1,1],
                                [0,0,1,1,1],
                                [0,0,1,1,1]], dtype=float)
        elif P==4:
            self.P = np.asarray([[1,1,0,0,0,0],
                                 [1,1,0,0,0,0],
                                 [0,0,1,1,1,0],
                                 [0,0,1,1,1,0],
                                 [0,0,1,1,1,0],
                                 [0,0,0,0,0,1]], dtype=float)

        self.P = self.P/np.sum(self.P, axis=1)[:,None]

            
        self.N_states = self.P.shape[0]
        
        self.L_V = L_V
        self.L_B = L_B
        self.L_E = L_E
        
        N_V_configs = self.N_states**self.L_V
        N_E_configs = self.N_states**self.L_E
        
        self.eigdecomp()
        self.init_dist = self.steady if init_dist is None else init_dist
        self.load_chain_configs()
        self.load_PV_PE(self.init_dist)
        
        print(f"Size of P(E|V):\t{self.P_E_cond_V.shape, }\n{time.time()-t0:.2f} seconds to initialize")
        
        
    def eigdecomp(self):
        wr, vr = np.linalg.eig(self.P)
        wl, vl = np.linalg.eig(self.P.T)

        steady = vl[:,0].real
        steady /= np.sum(steady)   
        self.steady = steady

        wr_argsort = np.argsort(np.abs(wr))[::-1] # args sorted from max to min (descending order)
        wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)

        self.lambd_1 = wr[wr_argsort][1].real

        R1 = vr.T[wr_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect
        L1 = vl.T[wl_argsort][1] # First sorts the columns (rows after transpose) and gets second eigenvect

        if self.verbose: print("IS R1 REAL? SUM IMAG: ", R1.imag.sum())
        if self.verbose: print("IS L1 REAL? SUM IMAG: ", L1.imag.sum())

        self.R1 = R1.real
        self.L1 = L1.real
        
        return
        
        
    def load_chain_configs(self):
        self.V_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_V))
        self.E_configs = list(itertools.product(np.arange(self.N_states), repeat=self.L_E))

        if len(self.V_configs)*len(self.E_configs) > 1e9: assert 1==0

        self.VR_configs = [V[-1] for V in self.V_configs]
        self.EL_configs = [E[0] for E in self.E_configs]
        
        return
    
    def load_PV_PE(self, init_dist=None):
        P_buffer = np.linalg.matrix_power(self.P, self.L_B+1)

        self.P_V = np.asarray([init_dist[V[0]]*np.prod([self.P[v_prv, v_nxt] for v_prv, v_nxt in zip(V[:-1], V[1:])]) for V in self.V_configs])
        self.P_E = np.asarray([init_dist[E[0]]*np.prod([self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs])
        P_E_free = np.asarray([np.prod([self.P[e_prv, e_nxt] for e_prv, e_nxt in zip(E[:-1], E[1:])]) for E in self.E_configs] )
        # Note: P_E_free above isn't a true probability distribution, since the initial state is free
        #P_E_joint_V = np.asarray([ [ pv*pe*P_buffer[VR_configs[iv], EL_configs[ie]] for iv, pv in enumerate(P_V)] for ie, pe in enumerate(P_E_free)])
        self.P_E_cond_V = np.asarray([ [ pe*P_buffer[self.VR_configs[iv], self.EL_configs[ie]] for iv, pv in enumerate(self.P_V)] for ie, pe in enumerate(P_E_free)])

        return

    
class PeriodicChain(MarkovChain):
    def __init__(self, L_V = 6, L_B = 1, L_E = 4, P=None, init_dist=None, verbose=False, doubleacc=True):
        self.verbose = verbose
        
        t0 = time.time()

        if isinstance(P, np.ndarray): self.P = P
        elif P==1:
            self.P =np.asarray([[0,1,0,1,0,0],
                                [0,0,1,0,0,0],
                                [0,0,0,1,0,0],
                                [1,0,0,0,1,0],
                                [0,0,0,0,0,1],
                                [1,0,0,0,0,0]], dtype=float)
            self.P = self.P/np.sum(self.P, axis=1)[:,None]
            
        elif P==2:
            self.P =np.asarray([
                    [0,0,1,2,0,0],
                    [0,0,2,3,0,0],
                    [0,0,0,0,2,1],
                    [0,0,0,0,1,1],
                    [3,1,0,0,0,0],
                    [1,2,0,0,0,0]], dtype=float)
            self.P = self.P/np.sum(self.P, axis=1)[:,None]

            
        self.N_states = self.P.shape[0]
        
        self.L_V = L_V
        self.L_B = L_B
        self.L_E = L_E
        
        N_V_configs = self.N_states**self.L_V
        N_E_configs = self.N_states**self.L_E
        
        self.eigdecomp()
        self.init_dist = self.steady if init_dist is None else init_dist
        self.load_chain_configs()
        self.load_PV_PE(self.init_dist, doubleacc=doubleacc)
        
        print(f"Size of P(E|V):\t{self.P_E_cond_V.shape, }\n{time.time()-t0:.2f} seconds to initialize")
        return
    
  
class TransientDisjointChain(DisjointChain):
    
    def __init__(self, L_V = 6, L_B = 1, L_E = 4, P=None, init_dist=None, verbose=False):
        self.verbose = verbose
        
        t0 = time.time()
        
        if isinstance(P, np.ndarray): self.P = P
        elif P==1:
            self.P = np.asarray([[1,1,0,0,0,0],
                                 [1,1,0,0,0,0],
                                 [0,0,1,1,0,0],
                                 [0,0,1,1,0,0],
                                 [1,2,3,1,5,6],
                                 [4,1,1,2,4,3]], dtype=float)

        self.P = self.P/np.sum(self.P, axis=1)[:,None]

            
        self.N_states = self.P.shape[0]
        
        self.L_V = L_V
        self.L_B = L_B
        self.L_E = L_E
        
        N_V_configs = self.N_states**self.L_V
        N_E_configs = self.N_states**self.L_E
        
        self.eigdecomp()
        self.init_dist = self.steady if init_dist is None else init_dist
        self.load_chain_configs()
        self.load_PV_PE(self.init_dist)
        
        print(f"Size of P(E|V):\t{self.P_E_cond_V.shape, }\n{time.time()-t0:.2f} seconds to initialize")
        
       