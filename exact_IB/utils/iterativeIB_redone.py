import numpy as np
import time
from tqdm.autonotebook import tqdm
import scipy.sparse

    
    
    
    
    
    
    
    
    
    
    
    
    
    


class iterativeIB(object):
    def __init__(self, 
                    P_V,
                    P_E,
                    P_E_cond_V,
                    analytic_kwargs={}, # Required if using analytical equation
                    verbose=False,
                 ):

        self.P_V = P_V
        self.P_E = P_E
        self.P_E_cond_V = P_E_cond_V
        self.counter = 0
        
    # For analytics
        VR = np.asarray(analytic_kwargs['V_configs']).T[-1] # Last states
        self.RV = analytic_kwargs['R1'][VR] # R1 is subleading right eigenvect. R(V) is <VR|R1>
        self.lambd_1 = analytic_kwargs['lambd_1']
        self.L_B = analytic_kwargs['L_B']
        self.P = analytic_kwargs.get('P', None)
        self.steady = analytic_kwargs.get('steady', None)

        self.verbose=verbose
    
    def calc_loss(self, beta, P_H_cond_V, P_H):
        P_V_cond_H = P_H_cond_V.T *self.P_V[:, None]/P_H[None, :]
        i_vh = np.nansum( (P_V_cond_H*P_H[None, :]).T * np.maximum(-1e10, np.log(P_H_cond_V/P_H[:, None])) )
        
        #def calc_Q_IB(MIB, MC, idx):
        ph = P_H.squeeze()
        phv = P_H_cond_V
        pvv = self.P 
        #ph = np.squeeze(phv @ self.P_V[:, None]))
        pvh = phv.T*self.P_V[:, None]/ph[None, :]
        #print(phv.shape, self.P_V.shape, pvv.shape, pvh.shape)
        phh = np.einsum('jl,kl,ki->ji', phv, pvv, pvh).T
        p_eh =  np.einsum('ij,ik->jk', self.P, pvh)   
        p_eh_joint = p_eh*ph[None,:]   
        
        i_eh = np.sum(p_eh_joint*np.log(p_eh/self.steady[:,None]))
        return i_vh - beta*i_eh
    
    def compute_hessian(self, P_H_cond_V, P_V, P_E_cond_V):
        P_V = np.squeeze(P_V)
        P_E_cond_H = np.einsum('ij,ki,i->jk', P_E_cond_V, P_H_cond_V, P_V)# sum_x p(y|x)p(h|x)p(x). P_{ij} = P(y_j | x_i) = p(y_j, h_k)
        P_H = (P_H_cond_V @ P_V[:, None]).squeeze()
        P_E_joint_V = P_E_cond_V * P_V[:, None]
        
        #print(P_E_joint_H.shape,  P_E_cond_V.shape, P_H_cond_V.shape, P_V.shape, P_H.shape)
        #zz = np.einsum('li,lj,l->ij', P_E_joint_V, P_E_joint_V, 1/P_E_cond_H[:,0])
        #zzz = P_V[:, None]*P_V[None, :]/P_H[0]
        #print(zz.shape, zzz.shape)

        hess_IXH = [(np.diag(P_V/P_H_cond_V[h, :]) - P_V[:, None]*P_V[None, :]/P_H[h]) for h in range(P_H_cond_V.shape[0])]# for  
        hess_IYH = [(np.einsum('li,lj,l->ij', P_E_joint_V, P_E_joint_V, 1/P_E_cond_H[:,h]) - P_V[:, None]*P_V[None, :]/P_H[h]) for h in range(P_H_cond_V.shape[0])]# for

        return P_E_joint_V, hess_IXH, hess_IYH, hess_IXH[0], hess_IYH[0] # single blocks
    
    def step_(self, beta, P_H_cond_V, P_H, longdouble, debug=False, record_loss=False):
        """
        Takes P(H|V), P(V), P(E|V), spits out new P(H|V) 
        """

        for i in range(self.iter_steps):
            P_V_cond_H = P_H_cond_V.T*self.P_V[:, None]/P_H[None, :]
            P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)
            # = p_E_cond_V = p(x_j | x_i ) * p(x_i|h)

        # Step 1: Calculate new P(H|V) using old P(H)
            x = beta*np.matmul(self.P_E_cond_V.T, np.maximum(-1e10, np.log(P_E_cond_H))).T
            
            P_H_cond_V = np.exp(x)
            
            P_H_cond_V = P_H_cond_V * P_H[:, None]
            
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0, keepdims=True)
            
            
            if not np.allclose(np.sum(P_H_cond_V, axis=0), np.ones(len(self.P_V))):
                P_H_cond_V *= np.nan
            
            

        # Step 2: Calculate new P(H)
            P_H = np.squeeze(np.matmul(P_H_cond_V, self.P_V[:, None]))
            #assert np.isclose(np.sum(P_H), 1)
            
            if record_loss=='always': 
                #print(self.counter)
                self.loss.append(self.calc_loss(beta, P_H_cond_V, P_H))
                #self.counter += 1


        P_V_cond_H = P_H_cond_V.T*self.P_V[:, None]/P_H[None, :]
        P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)   
        
        return P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H
              

    def init_run(self, C_H_init=1, init_encoder=None):
        if init_encoder=='uniform':
            P_H_cond_V = np.ones((C_H_init, len(self.P_V)))
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0)[None, :]
            #print('uniform init')
        elif init_encoder is not None:
            P_H_cond_V = init_encoder
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0)[None, :]
        else:
            P_H_cond_V = np.random.uniform(0,1,size=(C_H_init, len(self.P_V)))
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0)[None, :]
        return P_H_cond_V
        
    def run(self, iter_steps, 
                    betas, 
                    C_H,
                    longdouble=False,
                    randomness=0.1,
                    randomness_after_transition=1e-4,
                    init_encoder=None,
                    verbose=False,
                    save_pvh=False,
                    debug=False,
                    record_loss=False,
                    backtrack=False,
                    compute_hessian=False,
                    take_hessian_step=False): 
                                 
        if record_loss:     self.loss = []
        if compute_hessian: self.evals, self.evals_pre, self.loss_pre = [], [], []
            
        self.P_HV_list, self.P_HV_analytic_list  = [], []
        self.I_VH, self.I_EH = [], []
        self.C_H = C_H
        
        self.iter_steps=iter_steps
        self.betas = betas

        P_H_cond_V = self.init_run(C_H_init=C_H, init_encoder=init_encoder)
        
        if debug: print("P(H|V) init: \n", P_H_cond_V)

        t0 = time.time()
        
        if save_pvh:
            self.PHV_all = []
            
        pbar = tqdm(total=len(self.betas))
        
        hit_transition = False

        for b, beta in enumerate(self.betas):
            
            if compute_hessian=='before':
                _, hessX, hessY, _, _ = self.compute_hessian(P_H_cond_V, self.P_V, self.P_E_cond_V.T)
                
                for h in range(len(hessX)):
                    hesstot = -1*(np.asarray(hessX[h]) - beta*np.asarray(hessY[h]))

                    evals, evects = np.linalg.eig(hesstot.astype(float))
                    argsort = np.argsort(evals.real)[::-1] # highest firstv
                    evals = evals[argsort]
                    evects = evects[:, argsort]
                    self.evals_pre.append(evals[:5])
                    top_evect = evects[:,0]

                if take_hessian_step: 
                    P_H_cond_V_perturbed = P_H_cond_V + top_evect*take_hessian_step
                    P_H_perturbed = np.sum(P_H_cond_V_perturbed*self.P_V[None, :], axis=1)
                    self.loss_pre.append(self.calc_loss(beta, P_H_cond_V_perturbed, P_H_perturbed))
                    

            if randomness=='reset':
                P_H_cond_V = self.init_run(C_H_init=C_H, init_encoder=init_encoder)
                P_H_cond_V += 1e-4*np.random.uniform(0,1,size=P_H_cond_V.shape)
            elif hit_transition==True:
                P_H_cond_V += randomness_after_transition*np.random.uniform(0,1,size=P_H_cond_V.shape)
            else:
                P_H_cond_V += randomness*np.random.uniform(0,1,size=P_H_cond_V.shape)
                   
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0, keepdims=True)
            P_H = np.sum(P_H_cond_V*self.P_V[None, :], axis=1)
            
            if debug: print("P(H|V) init: \n", P_H_cond_V)
            if debug: print("P(H) init: \n", P_H)

            P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = self.step_(beta, P_H_cond_V, P_H, longdouble=longdouble, debug=debug, record_loss=record_loss)
            
            #print(P_E_cond_H.shape, self.P_E.shape, P_H.shape, P_V_cond_H.shape)
            
            i_vh = np.nansum( (P_V_cond_H*P_H[None, :]).T * np.maximum(-1e10, np.log(P_H_cond_V/P_H[:, None])) )
            i_eh = np.nansum( (P_E_cond_H*P_H[None, :]) * np.maximum(-1e10, np.log(P_E_cond_H/self.P_E[:,None])) )

            self.I_VH.append(i_vh)
            self.I_EH.append(i_eh)
            
            self.P_HV_list.append(P_H_cond_V)
            self.P_HV_analytic_list.append(self.calc_analytic(beta, P_V_cond_H, P_H))
            
            if record_loss=='beta': 
                self.loss.append(self.calc_loss(beta, P_H_cond_V, P_H))


            if compute_hessian:
                _,_,_, hessX, hessY = self.compute_hessian(P_H_cond_V, self.P_V, self.P_E_cond_V.T)
                
                hesstot = -1*(np.asarray(hessX) - beta*np.asarray(hessY))
                
                evals, evects = np.linalg.eig(hesstot.astype(float))
                argsort = np.argsort(evals.real)[::-1] # highest firstv
                evals = evals[argsort]
                evects = evects[:, argsort]
                self.evals.append(evals[:5])
                top_evect = evects[:,0]

                if take_hessian_step: 
                    P_H_cond_V_perturbed = P_H_cond_V + top_evect*take_hessian_step
                    P_H_perturbed = np.sum(P_H_cond_V_perturbed*self.P_V[None, :], axis=1)
                    self.loss.append(self.calc_loss(beta, P_H_cond_V_perturbed, P_H_perturbed))
                    
                    
                pbar.set_description(f'β = {beta:0.2f}, $\\lambda_1$ = {evals[0].real:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 
            else:
                pbar.set_description(f'β = {beta:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 
                
                
            #self.loss.append(i_vh - beta*i_eh)
            if np.any(np.isnan(P_H_cond_V)):
                break
                
            
            if compute_hessian=='after':
                _, hessX, hessY, _, _ = self.compute_hessian(P_H_cond_V, self.P_V, self.P_E_cond_V.T)
                
                for h in range(len(hessX)):
                    hesstot = -1*(np.asarray(hessX[h]) - beta*np.asarray(hessY[h]))

                    evals, evects = np.linalg.eig(hesstot.astype(float))
                    argsort = np.argsort(evals.real)[::-1] # highest firstv
                    evals = evals[argsort]
                    evects = evects[:, argsort]
                    self.evals_pre.append(evals[:2])
                    top_evect = evects[:,0]

                if take_hessian_step: 
                    P_H_cond_V = P_H_cond_V + top_evect*take_hessian_step
                    P_H = np.sum(P_H_cond_V*self.P_V[None, :], axis=1)
                    
        
            if verbose: print('Beta %u=%0.2f\tTime:%0.2f'%(b, beta, time.time()-t0))
            if save_pvh:
                self.PHV_all.append(P_H_cond_V)
                
            if hit_transition==False and i_vh>1e-5 and b>1:
                hit_transition = True
            
            #pbar.set_description(f'β = {beta:0.2f}, $\\lambda_1$ = {evals[0]:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 
            pbar.update(1)
            
            
        # Cast to arrays
        self.P_HV_list = np.asarray(self.P_HV_list)
        self.P_HV_analytic_list = np.asarray(self.P_HV_analytic_list)
        self.I_VH = np.asarray(self.I_VH)
        self.I_EH = np.asarray(self.I_EH)
            
        return 
        
        
    def calc_analytic(self, beta, P_V_cond_H, P_H):
        R_H = P_V_cond_H.T @ self.RV

        P_H_cond_V_analytic = np.exp(beta*((self.lambd_1.real)**(self.L_B+1))*(self.RV[:, None]@R_H[None,:])).T * P_H[:, None]
        P_H_cond_V_analytic = P_H_cond_V_analytic/np.sum(P_H_cond_V_analytic, axis=0, keepdims=True)
        
        return P_H_cond_V_analytic
    
    
    
    
    
    
    