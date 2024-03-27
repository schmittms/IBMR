
class agglomerativeIB(MC_iIB):
    def JS(self, p1, p2, pi=[0.5, 0.5]): 
        "jensen shannon divergence. Assume has shape [NH, ...]"
        pbar = pi[0]*p1 + pi[1]*p2
        Dkl1 = np.nansum( p1*np.maximum(-1e20, np.log(p1/pbar)), axis=1)
        Dkl2 = np.nansum( p2*np.maximum(-1e20, np.log(p2/pbar)), axis=1)
        return pi[0]*Dkl1 + pi[1]*Dkl2
            
          
    def step_(self, beta, P_H_cond_V, P_H, longdouble, debug=False, record_loss=False):
    #def step_(self, beta, P_H_cond_V):
        """
        Takes P(H|V), P(V), P(E|V), spits out new P(H|V) 
        """
    # Split
        C_H = len(P_H_cond_V)
        
        random_numbers = np.random.uniform(-0.5,0.5,size=P_H_cond_V.shape)
        P_H_cond_V_dupl = np.concatenate( [P_H_cond_V*(0.5 + self.alpha*random_numbers), 
                                           P_H_cond_V*(0.5 - self.alpha*random_numbers)], axis=0)
        P_H_dupl = np.sum(P_H_cond_V_dupl*self.P_V[None, :], axis=1)

    # Find IB fixed point
        for i in range(self.iter_steps):            
            P_V_cond_H = P_H_cond_V_dupl.T*self.P_V[:, None]/P_H_dupl[None, :]
            assert np.allclose(np.sum(P_V_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i

            P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)
            assert np.allclose(np.sum(P_E_cond_H, axis=0), np.ones(P_V_cond_H.shape[1])), i

            if np.any(P_E_cond_H==0):
                badE = np.any(P_E_cond_H==0, axis=1)
                badev = np.all(self.P_E_cond_V[badE]==0)
                #assert badev
                
            x = beta*np.matmul(self.P_E_cond_V.T, np.maximum(-1e10, np.log(P_E_cond_H))).T

            
            if longdouble:
                x = np.maximum(x, -11000)
                assert np.all(np.any(x > -11390., axis=0)),print( np.max(np.min(np.abs(x)/beta, axis=0)), np.any( x > -745., axis=0))
            else:
                x = np.maximum(x, -700)
                
            P_H_cond_V_dupl = np.exp(x)
            P_H_cond_V_dupl = P_H_cond_V_dupl * P_H_dupl[:, None]
            assert np.all( np.sum(P_H_cond_V_dupl, axis=0, keepdims=True)!=0 )
            
            P_H_cond_V_dupl = P_H_cond_V_dupl/np.sum(P_H_cond_V_dupl, axis=0, keepdims=True)

            P_H_dupl = np.squeeze(np.matmul(P_H_cond_V_dupl, self.P_V[:, None]))
            assert np.isclose(np.sum(P_H_dupl), 1)

            #P_V_cond_H = P_H_cond_V.T*self.P_V[:, None]/P_H[None, :]
            #P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)   


    # Calculate JS distance
        p_y_t1 = P_E_cond_H.T[:C_H] # Using notation from Slonim's thesis
        p_y_t2 = P_E_cond_H.T[C_H:]

        js = self.JS(p_y_t1, p_y_t2)
        char_to_split = js>self.dmin
        
        #print(f"js {js}:\tSPLITTING characters ", char_to_split)

        if np.any(char_to_split):
            #print("...splitting...")

            P_H_cond_V_old = P_H_cond_V_dupl[:C_H]
            P_H_cond_V_new = P_H_cond_V_dupl[C_H:]

            P_H_cond_V = np.vstack([P_H_cond_V_old, P_H_cond_V_new[char_to_split]])

        else:
            P_H_cond_V = P_H_cond_V_dupl[:C_H] + P_H_cond_V_dupl[C_H:]

        P_H = np.sum(P_H_cond_V*self.P_V[None, :], axis=1)
        P_V_cond_H = P_H_cond_V.T*self.P_V[:, None]/P_H[None, :]
        P_E_cond_H = np.matmul(self.P_E_cond_V, P_V_cond_H)   
        
        return P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H
              
        
    #def run(self, iter_steps=100, 
    #                betas=np.logspace(0,3,10), 
    #                alpha=0.2, # How much we should perturb P(H|V) when splitting
    #                dmin=1e-5): # This is the min distance between two pdists before splitting):
    def run(self, iter_steps, 
                    betas, 
                    alpha,
                    dmin,
                    max_CH=128,
                    longdouble=False,
                    init_encoder=None,
                    verbose=False,
                    save_pvh=False,
                    debug=False,
                    record_loss=False): 
        
        self.alpha=alpha
        self.dmin=dmin
        self.iter_steps=iter_steps
        self.betas = betas
        
        
        
        self.P_HV_list, self.P_HV_analytic_list  = [], []
        self.I_VH, self.I_EH = [], []

        P_H_cond_V = self.init_run(C_H_init=1, init_encoder=init_encoder)
        P_H_cond_V = P_H_cond_V/np.sum(P_H_cond_V, axis=0, keepdims=True)
        P_H = np.sum(P_H_cond_V*self.P_V[None, :], axis=1)
        
        pbar = tqdm(total=len(self.betas))

        for b, beta in enumerate(self.betas):
            #if self.verbose: print("Step %u: P(H|V) shape"%b, P_H_cond_V.shape)
            if P_H_cond_V.shape[0] > max_CH:
                P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = super().step_(beta, P_H_cond_V, P_H, longdouble=longdouble, debug=debug, record_loss=record_loss)
            else:
                P_H_cond_V, P_H, P_V_cond_H, P_E_cond_H = self.step_(beta, P_H_cond_V, P_H, longdouble=longdouble, debug=debug, record_loss=record_loss)
            
            
            i_vh = np.nansum( (P_V_cond_H*P_H[None, :]).T * np.maximum(-1e10, np.log(P_H_cond_V/P_H[:, None])) )
            i_eh = np.nansum( (P_E_cond_H*P_H[None, :]) * np.maximum(-1e10, np.log(P_E_cond_H/self.P_E[:,None])) )

            self.I_VH.append(i_vh)
            self.I_EH.append(i_eh)
            
            self.P_HV_list.append(P_H_cond_V)
            self.P_HV_analytic_list.append(self.calc_analytic(beta, P_V_cond_H, P_H))
            
            pbar.set_description(f'β = {beta:0.2f}, I = {i_vh:0.2e} (C_H={P_H_cond_V.shape[0]})') 
            pbar.update(1)
            
        # Cast to arrays
        self.P_HV_list = np.asarray(self.P_HV_list)
        self.P_HV_analytic_list = np.asarray(self.P_HV_analytic_list)
        self.I_VH = np.asarray(self.I_VH)
        self.I_EH = np.asarray(self.I_EH)
            
        return 
    