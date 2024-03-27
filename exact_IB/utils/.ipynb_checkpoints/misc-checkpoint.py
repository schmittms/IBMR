import numpy as np


def get_beta_crit_ind(I_VH, thresh=1e-5):
    diff = I_VH[1:]-I_VH[:-1]
    x = next( (n for n, k in enumerate(diff) if k>thresh) , np.nan) # Transition is between betas(x) and betas(x+1) 
    return x

def get_beta_crit_ind2(I_VH):
    diff = I_VH[1:]-I_VH[:-1]
    x = np.argmax(diff) # Transition is between betas(x) and betas(x+1) 
    return x

def calc_Q_IB(MIB, MC, idx):
    phv = MIB.P_HV_list[idx] # ph|v
    pvv = MC.P 
    ph = np.squeeze(np.matmul(phv, MIB.P_V[:, None]))
    pvh = phv.T*MIB.P_V[:, None]/ph[None, :]
    phh = np.einsum('jl,kl,ki->ji', phv, pvv, pvh).T
    p_eh =  np.einsum('ij,ik->jk', MC.P_E_cond_V.T, pvh)   
    p_eh_joint = p_eh*ph[None,:]   
    return phh, pvh, p_eh, p_eh_joint, ph

def calc_Q_IB2(MC, phv):
    pvv = MC.P 
    ph = np.squeeze(np.matmul(phv, MC.P_V[:, None]))
    pvh = phv.T*MC.P_V[:, None]/ph[None, :]
    phh = np.einsum('jl,kl,ki->ji', phv, pvv, pvh).T
    p_eh =  np.einsum('ij,ik->jk', MC.P_E_cond_V.T, pvh)   
    p_eh_joint = p_eh*ph[None,:]   
    return phh, pvh, p_eh, p_eh_joint, ph


def calc_Q_IB_alt(phv, pvv, pv):
    #phv = MIB.P_HV_list[idx] # ph|v
    #pvv = MC.P 
    ph = np.squeeze(np.matmul(phv, pv[:, None]))
    pvh = phv.T*pv[:, None]/ph[None, :]
    phh = np.einsum('jl,kl,ki->ji', phv, pvv, pvh).T
    p_eh =  np.einsum('ij,ik->jk', pvv, pvh)   
    p_eh_joint = p_eh*ph[None,:]   
    return phh, pvh, p_eh, p_eh_joint, ph

def calc_all_MI(phv, pvv, pv):
    Q, _, peh, peh_joint, ph = calc_Q_IB_alt(phv, pvv, pv)
    
    pvh_joint = phv.T*pv[:, None] #/ph[None, :]
    pvh = phv.T*pv[:, None]/(ph[None, :] + 1e-16)
    

    I_VH = np.nansum(pvh_joint*np.log(pvh / (pv[:, None] + 1e-16))) # I(X+, H)
    I_EH = np.nansum(peh_joint*np.log(peh / (pv[:, None] + 1e-16))) # I(X+, H)

    if np.any(np.isnan(Q)):
        return np.nan, np.nan, np.nan, np.nan

    wl, vl = np.linalg.eig(Q.T.astype(float))
    wl_argsort = np.argsort(np.abs(wl))[::-1] # args sorted from max to min (descending order)
    vl_sorted = vl.T[wl_argsort]
    steadyQ = vl_sorted[0].real
    steadyQ /= np.sum(steadyQ) 

    phh_joint = np.einsum('ij,i->ij', Q, ph)
    phh_reversed = Q

    H_HE = -np.sum(peh_joint * np.log(peh * ph[None, :] / pv[:, None]))
    H_HH = -np.sum(phh_joint * np.log(phh_reversed))

    diff = H_HH - H_HE

    H_Q = -np.sum(np.einsum('i,ij', steadyQ, Q*np.log(Q)))#\sum_i p 
    I_Q = -np.sum(steadyQ*np.log(steadyQ)) - H_Q # I(H+, H)
    
    
    return I_Q, I_EH, I_VH, diff