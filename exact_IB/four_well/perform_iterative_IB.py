import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import sys
if "/home/schmittms/exact_IB" not in sys.path: sys.path.append("/home/schmittms/exact_IB")
from utils import MarkovChain, iterativeIB, misc


if __name__=='__main__':
    ap = ArgumentParser()
    ap.add_argument('--n_steps', type=int, default=100)
    ap.add_argument('--nbins', type=int, default=100)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--x_max', type=float, default=1.0)
    ap.add_argument('--beta_max', type=float, default=100)
    ap.add_argument('--beta_min', type=float, default=100)
    ap.add_argument('--beta_steps', type=int, default=100)
    ap.add_argument('--beta_loglin', type=str, default=100)
    ap.add_argument('--iter_steps', type=int, default=100)
    ap.add_argument('--randomness_pre', type=float, default=100)
    ap.add_argument('--randomness', type=float, default=100)
    ap.add_argument('--C_H', type=int, default=100)
    ap.add_argument('--L_B', type=int, default=100)
    ap.add_argument('--steady', type=str, default=100)
    args = ap.parse_args()

    assert 1==0

    
    # Make Markov Chain
    xbins = np.linspace(-args.x_max, args.x_max, args.nbins+1)
    filename=f'P-ninit_100000-nsteps_100-nruns_2000-mu_{args.mu}-sigma_{args.sigma}-xmax_{args.x_max}-nbins_{args.nbins}'
    P = np.load(f'/home/schmittms/exact_IB/4Well/transition_matrices/{filename}.npy')
    
    MC = MarkovChain.MarkovChain(L_V = 1, L_B = args.L_B, L_E = 1, P=P, doubleacc=True)
    
    # Make optimizer
    acc = np.longdouble
    steady = np.ones(len(P))/len(P) if args.steady=='const' else MC.steady 
    
    MIB = iterativeIB.iterativeIB(MC.P_V.astype(acc), MC.P_E.astype(acc)[0], MC.P_E_cond_V.astype(acc),
                analytic_kwargs={'V_configs': MC.V_configs, 'R1': MC.R1.astype(acc), 'lambd_1': MC.lambd_1.astype(acc), 'L_B': MC.L_B,
                                 'P': MC.P,
                                 'steady': steady.astype(acc)},
                verbose=False)
    
    # Optimize
    if args.beta_loglin=='log':   betas_ = np.logspace(np.log10(args.beta_min), np.log10(args.beta_max), args.beta_steps)
    elif args.beta_loglin=='lin': betas_ = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    
    MIB.run(iter_steps=args.iter_steps, 
            betas=betas_, 
            C_H=args.C_H, 
            randomness=args.randomness_pre, 
            randomness_after_transition=args.randomness, 
            verbose=False, longdouble=True, record_loss=False)

    
    # Collect outputs
    p_hv_arr = np.asarray(MIB.P_HV_list)
    I_VH = np.asarray(MIB.I_VH)
    
    betas = MIB.betas

    I_EH = []
    Q_all = []
    for idx, beta in enumerate(betas):
        if idx>len(MIB.P_HV_list)-1: break
        Q, _, peh, peh_joint, _ = misc.calc_Q_IB(MIB, MC, idx)
        I_EH.append(np.sum(peh_joint*np.log(peh/MC.steady[:,None])))
        Q_all.append(Q)

    I_EH = np.asarray(I_EH)
    Q_all = np.asarray(Q_all)
    
    # Save
    filename = f'IB-nsteps_{args.n_steps}-mu_{args.mu}-sigma_{args.sigma}-xmax_{args.x_max}-beta_max_{args.beta_max}-beta_min_{args.beta_min}-beta_steps_{args.beta_steps}-beta_loglin_{args.beta_loglin}-iter_steps_{args.iter_steps}-randomness_pre_{args.randomness_pre}-randomness_{args.randomness}-C_H_{args.C_H}-LB_{args.L_B}-steady_{args.steady}-nbins_{args.nbins}'

    np.save(f'/home/schmittms/exact_IB/4Well/IB_results/{filename}.npy', {'p_hv': p_hv_arr,
                                                                                'P': P,  
                                                                               'I_VH': I_VH, 
                                                                               'I_EH': I_EH, 
                                                                               'Q_all': Q_all, 
                                                                               'betas': np.asarray(MIB.betas)}, allow_pickle=True)
