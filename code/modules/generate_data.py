from code import *

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from dynamical_models import Lorenz_63, Lorenz_96, oregonator, Adv_Dif_1D
from odelibrary import *
from computation_utils import file_to_dict
from time import time
from pdb import set_trace as bp

def generate_data(GD, rng_seed=1):
    """ Generate the true state, noisy observations and catalog of numerical simulations. """

    # initialization
    class xt:
        values = [];
        time = [];
    class yo:
        values = [];
        time = [];
    class catalog:
        analogs = [];
        successors = [];
        source = [];

    # test on parameters
    if GD.dt_states>GD.dt_obs:
        print('Error: GD.dt_obs must be bigger than GD.dt_states');
    if (np.mod(GD.dt_obs,GD.dt_states)!=0):
        print('Error: GD.dt_obs must be a multiple of GD.dt_states');

    # use this to generate the same data for different simulations
    np.random.seed(rng_seed);

    if (GD.model == 'Lorenz_63'):

        # 5 time steps (to be in the attractor space)
        x0 = np.array([8.0,0.0,30.0]);
        S = odeint(Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(Lorenz_63,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_test = S.shape[0];
        t_xt = np.arange(0,T_test,GD.dt_states);
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];

        # generate  partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(3),GD.sigma2_obs*np.eye(3,3),T_test);
        yo_tmp = S[t_xt,:]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;

        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];

        yo.time = xt.time;


        #generate catalog
        S =  odeint(Lorenz_63,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;

    elif (GD.model == 'Lorenz_96_MultiScale'):
        solver_type = 'hifiPlus' #RK45 w/ 1e-6 abstol, 1e-6 reltol, max_step=0.001
        delta_t = GD.dt_hifi_integration
        t_transient = 100
        t_data = GD.nb_loop_train
        # rng_seed, t_transient, t_data, delta_t, solver_type='default'
        # load solver dict
        solver_dict='../Config/solver_settings.json'
        foo = file_to_dict(solver_dict)
        solver_settings = foo[solver_type]

        ode = L96M(eps=GD.parameters.eps)
        f_ode = lambda t, y: ode.rhs(y,t)

        def simulate_traj(T1, T2):
            np.random.seed(rng_seed)
            t0 = 0
            u0 = ode.get_inits()
            print("Initial transients...")
            tstart = time()
            t_span = [t0, T1]
            t_eval = np.array([t0+T1])
            # sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, **solver_settings)
            sol = my_solve_ivp(ic=u0, f_rhs=f_ode, t_span=t_span, t_eval=t_eval, settings=solver_settings)
            print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')

            print("Integration...")
            tstart = time()
            u0 = np.squeeze(sol)
            t_span = [t0, T2]
            t_eval = np.arange(t0, T2+delta_t, delta_t)
            # sol = solve_ivp(fun=lambda t, y: self.rhs(t0, y0), t_span=t_span, y0=u0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
            # sol = solve_ivp(fun=f_ode, t_span=t_span, y0=u0, t_eval=t_eval, **solver_settings)
            sol = my_solve_ivp(ic=u0, f_rhs=f_ode, t_span=t_span, t_eval=t_eval, settings=solver_settings)
            print('took', '{:.2f}'.format((time() - tstart)/60),'minutes')
            return sol

        # make 1 long inv-meas trajectory
        x =  simulate_traj(T1=t_transient, T2=t_data)
        xt.time = np.arange(x.shape[0]) * delta_t
        xt.values = x[:,:ode.K]

    elif (GD.model == 'Lorenz_96_MultiScale_OLD'):
        ode = L96M(eps=GD.parameters.eps)

        # 5 time steps (to be in the attractor space)
        x0 = ode.get_inits()
        S = odeint(ode.rhs, x0, np.arange(0,5+0.000001,GD.dt_integration));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(ode.rhs, x0, np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration));
        T_test = S.shape[0];
        t_xt = np.arange(0,T_test,GD.dt_states);
        xt.time = t_xt*GD.dt_integration;

        # only store slow-system states
        xt.values = S[t_xt,:ode.K];

        # generate partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(ode.K), GD.sigma2_obs*np.eye(ode.K), T_test);
        yo_tmp = S[t_xt,:ode.K]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;
        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];
        yo.time = xt.time;


        # generate catalog
        S =  odeint(ode.rhs,S[S.shape[0]-1,:], np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(ode.K), GD.sigma2_catalog*np.eye(ode.K), T_train);
        catalog_tmp = S[:,:ode.K]+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;

    elif (GD.model == 'Lorenz_96'):

        # 5 time steps (to be in the attractor space)
        x0 = GD.parameters.F*np.ones(GD.parameters.J);
        x0[np.int(np.around(GD.parameters.J/2))] = x0[np.int(np.around(GD.parameters.J/2))] + 0.01;
        S = odeint(Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
        x0 = S[S.shape[0]-1,:];


        # generate true state (xt)
        S = odeint(Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
        T_test = S.shape[0];
        t_xt = np.arange(0,T_test,GD.dt_states);
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];


        # generate partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(GD.parameters.J),GD.sigma2_obs*np.eye(GD.parameters.J),T_test);
        yo_tmp = S[t_xt,:]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;
        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];
        yo.time = xt.time;


        # generate catalog
        S =  odeint(Lorenz_96,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(GD.parameters.J),GD.sigma2_catalog*np.eye(GD.parameters.J,GD.parameters.J),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;
    elif (GD.model == 'oregonator'):

        # 5 time steps (to be in the attractor space)
        x0 = np.array([4,1.1,4]);
        S = odeint(oregonator,x0,np.arange(0,10000+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        x0 = S[S.shape[0]-1,:];

        # generate true state (xt)
        S = odeint(oregonator,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        T_test = S.shape[0];
        t_xt = np.arange(0,T_test,GD.dt_states);
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];

        # generate  partial/noisy observations (yo)
        eps = np.random.multivariate_normal(np.zeros(3),GD.sigma2_obs*np.eye(3,3),T_test);
        yo_tmp = S[t_xt,:]+eps[t_xt,:];
        t_yo = np.arange(0,T_test,GD.dt_obs);
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;

        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];

        yo.time = xt.time;


        #generate catalog
        S =  odeint(oregonator,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration),args=(GD.parameters.alpha,GD.parameters.beta,GD.parameters.sigma));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states:,:]
        catalog.source = GD.parameters;
    elif (GD.model == 'Adv_Dif_1D'):
        class catalog:
            num_integration = [];
            true_solution = [];
            euler_integration = [];
            time = [];
        # 5 time steps (to be in the attractor space)
        x0 = np.array([GD.parameters.x0]);
        t0 = np.array([GD.parameters.t0]);
        t1 = np.array([GD.nb_loop_train]);
        # true solution
        t = np.arange(0,t1+0.000001,GD.dt_integration)
        true_sol = []
        for i in range(len(t)):
            true_sol.append(x0*np.exp(GD.parameters.w*t0)*np.exp(GD.parameters.w*t[i]))
        euler_sol = [x0]
        for i in range(1,len(t)):
            euler_sol.append(euler_sol[-1]+GD.dt_integration*GD.parameters.w*euler_sol[-1])

        r = ode(Adv_Dif_1D).set_integrator('zvode', method='bdf')
        r.set_initial_value(np.reshape(x0,(1,1)), t0).set_f_params(GD.parameters.w)
        t1 = GD.nb_loop_train
        dt = GD.dt_integration
        S = []
        while r.successful() and r.t < t1:
            r.integrate(r.t+dt)
            catalog.num_integration.append(r.y)
            catalog.time.append(r.t)
        catalog.num_integration = np.reshape(np.array(catalog.num_integration)[:,0,0],(len(catalog.num_integration),1))
        catalog.true_solution = np.array(true_sol)
        catalog.euler_integration = np.array(euler_sol)
    # reinitialize random generator number
    np.random.seed()
    return catalog, xt, yo;
