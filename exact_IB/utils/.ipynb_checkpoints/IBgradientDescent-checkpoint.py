import numpy as np
import time
from tqdm.autonotebook import tqdm
import scipy.sparse


class iterativeIBsparse(object):
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
        P_V_cond_H = (P_H_cond_V.T).multiply(self.P_V[:, None]).multiply(1/P_H[None, :])
        i_vh = np.nansum( (P_V_cond_H.multiply(P_H[None, :])).T * np.maximum(-1e10, np.log(P_H_cond_V/P_H[:, None])) )
        
        #def calc_Q_IB(MIB, MC, idx):
        phv = P_H_cond_V
        pvv = self.P 
        ph = np.squeeze(np.matmul(phv, self.P_V[:, None]))
        pvh = phv.T*self.P_V[:, None]/ph[None, :]
        #print(phv.shape, self.P_V.shape, pvv.shape, pvh.shape)
        phh = np.einsum('jl,kl,ki->ji', phv, pvv, pvh).T
        p_eh =  np.einsum('ij,ik->jk', self.P, pvh)   
        p_eh_joint = p_eh*ph[None,:]   
        
        i_eh = np.sum(p_eh_joint*np.log(p_eh/self.steady[:,None]))
        return i_vh - beta*i_eh
    
    def step_(self, beta, P_H_cond_V, P_H, longdouble, debug=False, record_loss=False):
        """
        Takes P(H|V), P(V), P(E|V), spits out new P(H|V) 
        """
    
        for i in range(self.iter_steps):
            P_V_cond_H = (P_H_cond_V.T).multiply(np.asarray(self.P_V)[:,None]).multiply(1/np.asarray(P_H).squeeze()[None,:])
            #assert np.allclose(np.sum(P_V_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i
            
            #if debug: print(f"Step {i}, SHAPEs P(V), P(H|V),\n", self.P_V.shape, P_H_cond_V.shape)
            
            P_E_cond_H = self.P_E_cond_V @ P_V_cond_H
            #assert np.allclose(np.sum(P_E_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i
            
            log_P_E_cond_H = P_E_cond_H.copy()
            log_P_E_cond_H = log_P_E_cond_H._with_data(np.log(log_P_E_cond_H.data), copy=True) # should automatically discard any infs
                        
            #x = np.maximum(x, -700)
            if debug: print(f"Step {i}\n\t x:\t {x}")

            #P_H_cond_V = np.exp(x)
        # Step 1: Calculate new P(H|V) using old P(H)
            x = beta*(self.P_E_cond_V.T @ log_P_E_cond_H).T
            
            print('x shape', x.shape)

            P_H_cond_V = x._with_data(np.exp(x.data), copy=True)
            P_H_cond_V = P_H_cond_V.multiply( np.asarray(P_H).squeeze()[:,None] )
            P_H_cond_V = P_H_cond_V.multiply(1/P_H_cond_V.sum(axis=0))#, keepdims=True)
            
            P_H = np.squeeze(np.asarray(P_H_cond_V @ self.P_V[:, None]))

            if debug: print(f"Step {i}\n\t P(H|V) post normalization:\n\t {P_H_cond_V}")

            if not np.allclose(np.asarray(P_H_cond_V.sum(axis=0)).squeeze(), np.ones(len(self.P_V))):
                P_H_cond_V *= np.nan
                print("Breaking...")
                break
            
            
        #print('PV:\t', self.P_V.shape, type(self.P_V))
        #print('PHV:\t', P_H_cond_V.shape, type(P_H_cond_V))
        #print('PH:\t', P_H.shape, type(P_H))
        
        P_V_cond_H = P_H_cond_V.T.multiply(self.P_V[:, None]).multiply(1/P_H[None, :])
        P_E_cond_H = self.P_E_cond_V @ P_V_cond_H
        
        return P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H
    
    
    def step_highbeta(self, beta, P_H_cond_V, P_H, longdouble, debug=False, record_loss=False):
        """
        Takes P(H|V), P(V), P(E|V), spits out new P(H|V) 
        """
            
        for i in range(self.iter_steps):
            P_V_cond_H = (P_H_cond_V.T).multiply(np.asarray(self.P_V)[:,None]).multiply(1/np.asarray(P_H).squeeze()[None,:])
            
            P_E_cond_H = self.P_E_cond_V @ P_V_cond_H
            
            #print(np.min(P_E_cond_H.data))
            #print(np.min(np.log(P_E_cond_H.data)))
            log_P_E_cond_H = P_E_cond_H.copy()
            #log_P_E_cond_H = log_P_E_cond_H._with_data(np.log(np.abs(log_P_E_cond_H.data)), copy=True) # should automatically discard any infs. Here the nans occur: take abs to prevent
            log_P_E_cond_H = log_P_E_cond_H._with_data(np.log( np.maximum(1e-60, log_P_E_cond_H.data)), copy=True) # should automatically discard any infs. Here the nans occur
            
            #print('log eh', np.any(np.isnan(log_P_E_cond_H.toarray())))
                        
            logit_hv = beta*(self.P_E_cond_V.T @ log_P_E_cond_H).T # shape (H, V)
            
            #print('log hv', np.any(np.isnan(logit_hv.toarray())))

            #exp_hv = (logit_hv._with_data(np.exp(logit_hv.data), copy=True)).multiply(np.asarray(P_H).squeeze())#.sum(axis=0)
            P_H = np.asarray(P_H).squeeze()
            P_H_cond_V = np.zeros(P_H_cond_V.shape, dtype=P_H_cond_V.dtype)
            for i in range(logit_hv.shape[0]):
                dlogit = np.asarray(logit_hv - logit_hv.toarray()[i]) # shape [H, V]
                
                #print(np.any(np.isnan(dlogit)))

                dlogit = dlogit*(P_H[:, None]/P_H[i])
                
                #print(np.any(np.isnan(dlogit)))
                
                #print(np.any(np.sum(np.exp(dlogit), axis=0)==0), )
                
                new_p = 1/np.sum(np.exp(dlogit), axis=0)
                
                #print(np.any(new_p==0), np.any(np.isnan(new_p)))
                
                P_H_cond_V[i] = new_p#1/np.sum(np.exp(dlogit), axis=0)
                
            P_H_cond_V = scipy.sparse.csr.csr_matrix(P_H_cond_V)

            P_H = np.squeeze(np.asarray(P_H_cond_V @ self.P_V[:, None]))

            
            if debug: print(f"Step {i}\n\t P(H|V) post normalization:\n\t {P_H_cond_V}")

            #if not np.allclose(np.asarray(P_H_cond_V.sum(axis=0)).squeeze(), np.ones(len(self.P_V))):
            #    print(np.asarray(P_H_cond_V.sum(axis=0)).squeeze())
                #P_H_cond_V *= np.nan
            #    print("Breaking...")
            #    return P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H
            
            
        #print('PV:\t', self.P_V.shape, type(self.P_V))
        #print('PHV:\t', P_H_cond_V.shape, type(P_H_cond_V))
        #print('PH:\t', P_H.shape, type(P_H))
        
        P_V_cond_H = P_H_cond_V.T.multiply(self.P_V[:, None]).multiply(1/P_H[None, :])
        P_E_cond_H = self.P_E_cond_V @ P_V_cond_H
        
        return P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H
              

    def init_run(self, C_H_init=1, init_encoder=None):
        if init_encoder is not None:
            P_H_cond_V = init_encoder
            P_H_cond_V = P_H_cond_V.multiply(1/P_H_cond_V.sum( axis=0))#[None, :]
        else:
            P_H_cond_V = scipy.sparse.csr.csr_matrix(np.random.uniform(0,1,size=(C_H_init, len(self.P_V))))
            P_H_cond_V = P_H_cond_V.multiply(P_H_cond_V.sum( axis=0))#[None, :]
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
                    use_highbeta_step=False): 
                                 
        if record_loss:
            self.loss = []
            
        self.P_HV_list, self.P_HV_analytic_list  = [], []
        self.I_VH, self.I_EH = [], []
        self.C_H = C_H
        self.uniform_init = uniform_init
        
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
            #pbar.set_description(f'β = {beta:0.2f}') 

            if randomness=='reset':
                P_H_cond_V = self.init_run(C_H_init=C_H, init_encoder=init_encoder)
            elif hit_transition==True:
                P_H_cond_V += scipy.sparse.csr.csr_matrix(randomness_after_transition*np.random.uniform(0,1,size=P_H_cond_V.shape))
            else:
                P_H_cond_V += scipy.sparse.csr.csr_matrix(randomness*np.random.uniform(0,1,size=P_H_cond_V.shape))
                   
            P_H_cond_V = P_H_cond_V.multiply(P_H_cond_V.sum(axis=0))#, keepdims=True)
            P_H = (P_H_cond_V.multiply(self.P_V[None, :])).sum(axis=1)
            
            if debug: print("P(H|V) init: \n", P_H_cond_V)
            if debug: print("P(H) init: \n", P_H)
            
            if use_highbeta_step: P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = self.step_highbeta(beta, P_H_cond_V, P_H, longdouble=longdouble, debug=debug, record_loss=record_loss)
            else:                 P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = self.step_(beta, P_H_cond_V, P_H, longdouble=longdouble, debug=debug, record_loss=record_loss)

            P_H = np.asarray(P_H).squeeze()
                   
            logv = P_H_cond_V.multiply(1/P_H[:, None])
            logv = logv._with_data(np.log(logv.data), copy=True) # should automatically discard any infs
            loge = P_E_cond_H.multiply(self.P_E[:,None])
            loge = loge._with_data(np.log(loge.data), copy=True) # should automatically discard any infs
            
            i_vh = (P_V_cond_H.multiply(P_H[None, :]).T ).multiply(logv)
            i_eh = (P_E_cond_H.multiply(P_H[None, :])).multiply(loge )
            i_vh = i_vh.sum()
            i_eh = i_eh.sum()

            self.I_VH.append(i_vh)
            self.I_EH.append(i_eh)
            
            self.P_HV_list.append(P_H_cond_V)
            self.P_HV_analytic_list.append(self.calc_analytic(beta, P_V_cond_H, P_H))
            
            #self.loss.append(i_vh - beta*i_eh)
            if np.any(np.isnan(P_H_cond_V.data)):
                break
                   
        
            if verbose: print('Beta %u=%0.2f\tTime:%0.2f'%(b, beta, time.time()-t0))
            if save_pvh:
                self.PHV_all.append(P_H_cond_V)
                
            if hit_transition==False and i_vh>1e-5 and b>1:
                hit_transition = True
            
            pbar.set_description(f'β = {beta:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 
            pbar.update(1)
            
            
        # Cast to arrays
        #self.P_HV_list = np.asarray(self.P_HV_list)
        #self.P_HV_analytic_list = np.asarray(self.P_HV_analytic_list)
        self.I_VH = np.asarray(self.I_VH)
        self.I_EH = np.asarray(self.I_EH)
            
        return 
        
        
    def calc_analytic(self, beta, P_V_cond_H, P_H):
        R_H = P_V_cond_H.T @ self.RV

        P_H_cond_V_analytic = np.exp(beta*((self.lambd_1.real)**(self.L_B+1))*(self.RV[:, None]@R_H[None,:])).T * P_H[:, None]
        P_H_cond_V_analytic = P_H_cond_V_analytic/np.sum(P_H_cond_V_analytic, axis=0, keepdims=True)
        
        return P_H_cond_V_analytic
    
    
    
    
    
    
    
    
    
    
    


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
            #assert np.allclose(np.sum(P_V_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i
            
            #if debug: print(f"Step {i}, SHAPEs P(V), P(H|V),\n", self.P_V.shape, P_H_cond_V.shape)
            
            P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)
            #assert np.allclose(np.sum(P_E_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i

        # Step 1: Calculate new P(H|V) using old P(H)
            x = beta*np.matmul(self.P_E_cond_V.T, np.maximum(-1e10, np.log(P_E_cond_H))).T
            
            #x = np.maximum(x, -700)
            if debug: print(f"Step {i}\n\t x:\t {x}")

            P_H_cond_V = np.exp(x)
            
            if debug: print(f"Step {i}\n\t P(H|V):\t {P_H_cond_V}")
            if debug: print(f"If there are problems: are your sectors distinct, and does steady state of P not have prob. mass in all sectors?")

            """
            The sum (matmul) gets values up to ~5. The min value that the exponent can have is np.log(np.nextafter(0, 1))
            approx -745. So that means you'll start to have issues once beta ~ 150.
            """
            
            P_H_cond_V = P_H_cond_V * P_H[:, None]
            
            #if longdouble:
            #    assert np.all(np.any( x > -11390., axis=0)), print( np.max(np.min(np.abs(x)/beta, axis=0)), np.any( x > -745., axis=0) )
            #else:
            #    abc =1
            #    #assert np.all(np.any( x > -745., axis=0)), print( np.max(np.min(np.abs(x)/beta, axis=0)), np.any( x > -745., axis=0) )
            #assert np.all( np.sum(P_H_cond_V, axis=0, keepdims=True)!=0 )
            
            #bad_V = np.sum(P_H_cond_V, axis=0) == 0
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0, keepdims=True)
            
            #P_H_cond_V[:, bad_V] = 1/len(P_H) 
            
            if debug: print(f"Step {i}\n\t P(H|V) post normalization:\n\t {P_H_cond_V}")

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
            
            if compute_hessian:
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
                #print(self.counter)
                #print('saving loss')
                self.loss.append(self.calc_loss(beta, P_H_cond_V, P_H))
            
            if compute_hessian:
                _,_,_, hessX, hessY = self.compute_hessian(P_H_cond_V, self.P_V, self.P_E_cond_V.T)
                
                hesstot = -1*(np.asarray(hessX) - beta*np.asarray(hessY))
                
                evals, evects = np.linalg.eig(hesstot.astype(float))
                argsort = np.argsort(evals.real)[::-1] # highest firstv
                evals = evals[argsort]
                evects = evects[:, argsort]
                self.evals.append(evals[:2])
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class iterativeIBkicks(iterativeIB):
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
                    take_hessian_step=False,
                    kick_betas=[],
                    kick_magnitudes=[],
                    kick_directions=[]): 
                                 
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
            
            for k, kb in enumerate(kick_betas):
                if beta<kb and self.betas[b+1]>kb:
                    print("KICK!")
                    P_H_cond_V = P_H_cond_V + kick_magnitudes[k]*kick_directions[k]
                
            self.P_HV_list.append(P_H_cond_V)

            if compute_hessian:
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
                self.evals.append(evals[:2])
                top_evect = evects[:,0]

                if take_hessian_step: 
                    P_H_cond_V_perturbed = P_H_cond_V + top_evect*take_hessian_step
                    P_H_perturbed = np.sum(P_H_cond_V_perturbed*self.P_V[None, :], axis=1)
                    self.loss.append(self.calc_loss(beta, P_H_cond_V_perturbed, P_H_perturbed))
                    
                    
                pbar.set_description(f'β = {beta:0.2f}, $\\lambda_1$ = {evals[0]:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 
            else:
                pbar.set_description(f'β = {beta:0.2f}, I = {i_vh:0.2e} (hit={hit_transition})') 


            #self.loss.append(i_vh - beta*i_eh)
            if np.any(np.isnan(P_H_cond_V)):
                break
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


class MC_iIB_randomInits(iterativeIB):
    def run(self, iter_steps=100, 
                    betas=np.logspace(0,3,10), 
                    C_H=3,
                    longdouble=False): 
        
        self.C_H = C_H
        
        self.iter_steps=iter_steps
        self.betas = betas

        t0 = time.time()

        betas = np.logspace(-0.2, 0.2, 15)

        for b, beta in enumerate(self.betas):
            
            P_H_cond_V = self.init_run(C_H_init=C_H)
            P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0, keepdims=True)
            P_H = np.sum(P_H_cond_V*self.P_V[None, :], axis=1)
            
            P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = self.step_(beta, P_H_cond_V, P_H, longdouble=longdouble)
            
            i_vh = np.nansum( (P_V_cond_H*P_H[None, :]).T * np.maximum(-1e10, np.log(P_H_cond_V/P_H[:, None])) )
            i_eh = np.nansum( (P_E_cond_H*P_H[None, :]) * np.maximum(-1e10, np.log(P_E_cond_H/self.P_E[:,None])) )

            self.I_VH.append(i_vh)
            self.I_EH.append(i_eh)
            
            self.P_HV_list.append(P_H_cond_V)
            self.P_HV_analytic_list.append(self.calc_analytic(beta, P_V_cond_H, P_H))
                   
        
            print('Beta %u=%0.2f\tTime:%0.2f'%(b, beta, time.time()-t0))
            
            
        # Cast to arrays
        self.P_HV_list = np.asarray(self.P_HV_list)
        self.P_HV_analytic_list = np.asarray(self.P_HV_analytic_list)
        self.I_VH = np.asarray(self.I_VH)
        self.I_EH = np.asarray(self.I_EH)
            
        return 

    
