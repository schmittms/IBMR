import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


if __name__=='__main__':
    ap = ArgumentParser()
    ap.add_argument('--n_init_cond', type=int, default=10000)
    ap.add_argument('--n_steps', type=int, default=100)
    ap.add_argument('--n_runs', type=int, default=1000)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--x_max', type=float, default=1.0)
    ap.add_argument('--n_bins', type=int, default=100)
    args = ap.parse_args()

    def F(x):
        return -0.1*(8*x*(x**2 - 1)*(1 + args.mu*(x-1) + x)*(-1+args.mu+ (1+args.mu)*x)*(-1 - args.mu**2 + (1 + args.mu)**2*x**2))/(-1 + args.mu)**4


    n_init_cond = args.n_init_cond//100
    n_steps = args.n_steps

    dt = 0.002
    σ = args.sigma

    T = np.arange(n_steps)*dt


    for it in range(100):
        init_x = np.random.uniform(low=-args.x_max, high=args.x_max, size=(n_init_cond)) 
        x = np.zeros((n_init_cond, n_steps))
        x[:, 0] = init_x

        n_runs = args.n_runs
        x_finals = []
        x_all = []

        init0 = []
        trajs = []
        pbar = tqdm(total=n_runs)


        for n in range(n_runs):
            for t in np.arange(1, n_steps):
                x[:, t] = x[:, t-1] + F(x[:, t-1])*dt + σ*np.random.randn(n_init_cond)*np.sqrt(dt)

            x_all.append(x[0,:].copy())
            init0.append(x[0,-1].copy())
            x_finals.append([x[:,0].copy(), x[:,-1].copy()])
            pbar.update(1)


        trajs = np.asarray(x_finals).swapaxes(0,1).reshape(2,-1)

        x0_dat = trajs[0,:,None]
        xt_dat = trajs[1,:,None]

        xbins = np.linspace(-args.x_max, args.x_max, args.n_bins+1)

        H, _, _ = np.histogram2d(x0_dat.ravel(), xt_dat.ravel(), bins=[xbins, xbins])

        if it==0:
            H_sum = H
        else:
            H_sum += H
        
 
    P = H_sum/np.sum(H_sum, axis=1)[:,None]
    
    filename = f'P-ninit_{args.n_init_cond}-nsteps_{args.n_steps}-nruns_{args.n_runs}-mu_{args.mu}-sigma_{args.sigma}-xmax_{args.x_max}-nbins_{args.n_bins}'
    
    np.save(f'/home/schmittms/exact_IB/4Well/transition_matrices/{filename}.npy', P)
