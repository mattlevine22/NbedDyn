import sys, os
# local_path = "/homes/s17ouala/Bureau/Sanssauvegarde/NbedDyn-main"
local_path = '/Users/matthewlevine/code_projects/NbedDyn'
sys.path.append(os.path.join(local_path, 'code/modules'))
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
import torch
from odelibrary        import *
from dynamical_models   import *
from generate_data      import *
from NbedDyn_componentwise2   import *
# from NbedDyn_componentwise   import *
from stat_functions     import *
from computation_utils import *
from rf import RF
import pandas as pd
import seaborn as sns

from pdb import set_trace as bp

def main(eps_exp = -7, N_lat=2):
    # eps_exp = -2 # -7 is very fast mixing, -1 is very slow and is the first integer where markovian dramatically breaks.

    foo_eps = 2**(eps_exp)
    ode = L96M(slow_only=True, share_gp=True, add_closure=False)

    # run the data generation
    train_path = os.path.join(local_path,'data/X_train_L96MS_2e{}.npy'.format(eps_exp))
    test_path = os.path.join(local_path,'data/X_test_L96MS_2e{}.npy'.format(eps_exp))
    class GD:
        model = 'Lorenz_96_MultiScale'
        class parameters:
            K = 9
            eps = foo_eps
    #         sigma = 10.0
    #         rho = 28.0
    #         beta = 8.0/3
        dt_integration = 0.001 # integration time (timestep used to downsample data and train the models. can change this and re-use original L96 data)
        dt_hifi_integration = 0.0001 # output time for data generation (if you change this, you must generate new data!)
        dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
        dt_obs = 8# number of integration times between consecutive observations (for yo)
        var_obs = np.arange(parameters.K) # indices of the observed variables
        nb_loop_train = 2200 # size of the catalog
        # nb_loop_test = 100 # size of the true state and noisy observations
        sigma2_catalog = 0.0 # variance of the model error to generate the catalog
        sigma2_obs = 0.0 # variance of the observation error to generate observation

    #### load (or generate) data
    try:
        X_train_chaos_all = np.load(train_path)
        # X_test  = np.load(test_path)
    except:
        catalog, xt, yo = generate_data(GD,rng_seed=1)
        np.save(train_path, xt.values)
        X_train_chaos_all = np.load(train_path)

        # catalog, xt_test, yo = generate_data(GD,rng_seed=2)
        # np.save(test_path, xt_test.values)
        # X_test  = np.load(test_path)

    #### downsample to specifc training sampling rate
    X_train_chaos_all = subsample(x=X_train_chaos_all, dt_given=GD.dt_hifi_integration, dt_subsample=GD.dt_integration)

    N_train = int(85/GD.dt_integration)
    N_test_short = int(5/GD.dt_integration)
    N_test_long = int(10/GD.dt_integration) - 1

    X_train_chaos = X_train_chaos_all[:N_train]
    X_test = X_train_chaos_all[N_train:(N_train + N_test_long)]

    #####
    d_obs      = 9
    X_train    = X_train_chaos[:,:d_obs]#[:-1,:1]
    Grad_t     = gradient(X_train, dt=GD.dt_integration, method='spline') # can try method='spline' for better accuracy
    Batch_size = X_train.shape[0]
    nb_batch   = int(X_train.shape[0]/Batch_size)
    X_train    = X_train.reshape(nb_batch,Batch_size,d_obs)
    Grad_t     = Grad_t.reshape(nb_batch,Batch_size,d_obs)
    x_grid     = np.arange(np.min(X_train), np.max(X_train), step=0.1)

    # setup nbedDyn
    d_component = 1
    seed = 0
    params = {}
    params['use_f0']             = True
    params['f0']                 = ode.rhs
    params['seed']               = seed
    params['transition_layers']  = 1
    params['bi_linear_layers']   = N_lat+d_component # acts component-wise
    params['dim_hidden_linear']  = N_lat+d_component  # acts component-wise
    params['dim_input']          = d_obs # used to collect entire observation vector
    params['dim_latent']         = N_lat  # acts component-wise
    params['dim_observations']   = d_component # used to operate on single component
    params['dim_Embedding']      = N_lat+d_component  # acts component-wise
    params['ntrain']             = [30000,1000]
    params['dt_integration']     = GD.dt_integration
    params['pretrained']         = False
    params['nb_batch']           = nb_batch
    params['Batch_size']         = Batch_size
    params['get_latent_train']   = False
    params['path']               = '../output/L96MS_eps_2e{}/Latent{}/'.format(eps_exp, N_lat)
    params['file_name']          = 'NbedDyn_L96_dim_'+str(params['dim_Embedding'])+'_seed_'+str(params['seed'])

    # create output path
    os.makedirs(params['path'], exist_ok=True)
    fig_path_kde = os.path.join(params['path'], 'invariant_measure_KDE')

    ##### plot residual discrepancy between f0 and fdag for training set
    fig_path = os.path.join(params['path'], 'L96_residuals')
    f0_train = ode.rhs(torch.FloatTensor(X_train).T,0).T.data.numpy()
    plt.scatter(X_train.reshape(-1), (Grad_t - f0_train).reshape(-1))
    plt.savefig(fig_path)
    plt.close()

    ##### first do f0only model
    params['file_name'] += '_f0'
    model_f0, modelRINN_f0 = get_NbedDyn_model(params, use_f0=True, doMark=False, doNonMark=False)
    train_NbedDyn_model_L96MS(params,model_f0,modelRINN_f0,X_train,Grad_t)

    # generate test predictions by continuing the training set
    y0 = np.squeeze(X_train[-1,-1:,:])
    # pred_f0 = [torch.cat((torch.from_numpy(X_train).float()[-1,-1:,:], modelRINN_f0.Dyn_net.y_aug[-1,-1:,:]), dim=1)]
    # for i in range(N_test_long):
    #     pred_f0.append(modelRINN_f0(pred_f0[-1],params['dt_integration'])[0])

    # generate test preds again, but using implicit solver
    settings= {'method': 'RK45'}
    t_eval = GD.dt_integration*np.arange(-1, N_test_long+1)
    t_span = [t_eval[0], t_eval[-1]]
    sol_f0 = my_solve_ivp(y0[:ode.K], lambda t, y: ode.rhs(y,t), t_eval, t_span, settings)

    ####### plot KDE statistics
    df = pd.DataFrame({'True': X_test[:N_test_long].reshape(-1),
                       # 'f0': torch.stack(pred_f0)[:N_test_long,:,:d_obs].data.numpy().reshape(-1),
                       'f0 RK45 adaptive': sol_f0[:N_test_long,:d_obs].reshape(-1)})
                       # 'f0 + Mark': torch.stack(pred_f0_mark)[:N_test_long,:,:d_obs].data.numpy().reshape(-1),
                       # 'f0 + Mark + NonMark': torch.stack(pred_f0_mark_nonMark)[:N_test_long,:,:d_obs].data.numpy().reshape(-1)})
    sns.kdeplot(data=df)
    plt.savefig(fig_path_kde)
    plt.close()


    ##### now do f0 + markovian w/ random features
    rf = RF(do_normalization=True, lam_rf=1e-3)
    x_input = X_train.reshape(1,-1)
    y_output = (Grad_t - f0_train).reshape(-1)
    rf.fit(x_input, y_output)
    m = rf.predict(x_grid.reshape(1,-1), use_torch=False)
    fig_path = os.path.join(params['path'], 'L96_residuals_Mark_RF')
    n_t1 = int(1/GD.dt_integration)
    n_burnin = n_t1
    plt.scatter(x_input, y_output, label='Residual derivative')
    plt.scatter(x_grid, m, label='Markovian fit')
    plt.plot(X_train[0,n_burnin:n_burnin+n_t1,0], (Grad_t - f0_train)[0,n_burnin:n_burnin+n_t1,0], color='magenta', linewidth=5, label='Derivative trajectory (T=1)')
    plt.legend()
    plt.savefig(fig_path)
    plt.close()

    ##### do f0 + markovian w/ nbedDyn
    # params['ntrain'] = [500,1000]
    # params['file_name'] += '_Mark'
    # model, modelRINN = get_NbedDyn_model(params, use_f0=True, doMark=True, doNonMark=False)
    # model_f0_mark, modelRINN_f0_mark, aug_inp_data_f0_mark = train_NbedDyn_model_L96MS(params,model,modelRINN,X_train,Grad_t, nm='f0+Mark')

    # plot the structure of the learnt markovian term
    # fig_path = os.path.join(params['path'], 'L96_residuals_Mark')
    # plt.scatter(X_train.reshape(-1), (Grad_t - f0_train).reshape(-1))
    # m = np.squeeze(model_f0_mark.f_nn_Mark(torch.FloatTensor(x_grid[:,None])).data.numpy())
    # plt.scatter(x_grid, m)
    # plt.savefig(fig_path)
    # plt.close()

    # generate test predictions by continuing the training set
    # y0 = np.squeeze(modelRINN_f0_mark.Dyn_net.y_aug[-1,-1:,:])
    # pred_f0_mark = [torch.cat((torch.from_numpy(X_train).float()[-1,-1:,:], modelRINN_f0_mark.Dyn_net.y_aug[-1,-1:,:]), dim=1)]
    # for i in range(N_test_long):
    #     pred_f0_mark.append(modelRINN_f0_mark(pred_f0_mark[-1],params['dt_integration'])[0])

    # integrate with solve_ivp
    ode.set_predictor(rf.predict)
    ode.add_closure = True
    sol_Mark = my_solve_ivp(y0[:ode.K], lambda t, y: ode.rhs(y,t), t_eval, t_span, settings)

    # compute NbedDyn loss
    params['file_name'] += '_Mark'
    # ode.debug = True
    params['f0'] = ode.rhs
    model_f0_mark, modelRINN_f0_mark = get_NbedDyn_model(params, use_f0=True, doMark=False, doNonMark=False)
    train_NbedDyn_model_L96MS(params, model_f0_mark, modelRINN_f0_mark, X_train, Grad_t)

    ####### plot KDE statistics
    df = pd.DataFrame({'True': X_test[:N_test_long].reshape(-1),
                       # 'f0': torch.stack(pred_f0)[:N_test_long,:,:d_obs].data.numpy().reshape(-1),
                       'f0 RK45 adaptive': sol_f0[:N_test_long,:d_obs].reshape(-1),
                       'f0 + Mark RK45 adaptive': sol_Mark[:N_test_long,:d_obs].reshape(-1)})
                       # 'f0 + Mark': torch.stack(pred_f0_mark)[:N_test_long,:,:d_obs].data.numpy().reshape(-1)})
                       # 'f0 + Mark + NonMark': torch.stack(pred_f0_mark_nonMark)[:N_test_long,:,:d_obs].data.numpy().reshape(-1)})
    sns.kdeplot(data=df)
    plt.savefig(fig_path_kde)
    plt.close()

    ###### plot predictions vs Test
    plot_dir = os.path.join(params['path'], 'trajectory_forecast_test')
    os.makedirs(plot_dir, exist_ok=True)
    for ind in range(d_obs):
        fig_path = os.path.join(plot_dir, 'state{}'.format(ind))
        fig, axs = plt.subplots(ncols=1, figsize=(15, 10))
        axs.plot(X_test[:N_test_short,ind], label='True')
        axs.plot(sol_f0[:N_test_short,ind], label='f0')
        axs.plot(sol_Mark[:N_test_short,ind], label='f0 + Mark')
        axs.legend()
        plt.savefig(fig_path)
        plt.close()

    ##### now do f0 + markovian + nonMarkovian
    # params['ntrain'] = [1000,1000]
    params['ntrain'] = [320,100]
    params['file_name'] += '_nonMark'

    model, modelRINN = get_NbedDyn_model(params, use_f0=True, doMark=False, doNonMark=True)
    model_f0_mark_nonMark, modelRINN_f0_mark_nonMark, aug_inp_data_f0_mark_nonMark = train_NbedDyn_model_L96MS(params,model,modelRINN,X_train,Grad_t, nm='f0+Mark+nonMark')

    # plot the inferred dynamics
    foo = np.squeeze(modelRINN_f0_mark_nonMark.Dyn_net.y_aug.data.numpy()) # 10000 x D

    # observed states
    fig_path = os.path.join(params['path'], 'observed_dynamics_nonMark_training_fit')
    fig, axs = plt.subplots(ncols=d_obs, figsize=(200, 10))
    for ind in range(d_obs):
        axs[ind].plot(X_train[:,ind], label='True')
        axs[ind].plot(foo[:,ind], label='Trained Output')
        axs[ind].legend()
    plt.savefig(fig_path)
    plt.close()

    # hidden states
    fig_path = os.path.join(params['path'], 'hidden_dynamics_nonMark_training_fit')
    fig, axs = plt.subplots(ncols=1, figsize=(20, 10))
    for ind in range(d_obs, foo.shape[1]):
        axs.plot(foo[:,ind], label='State {}'.format(ind-d_obs))
    axs.legend()
    plt.savefig(fig_path)
    plt.close()

    # residual derivative
    plot_dir = os.path.join(params['path'], 'L96_residuals_nonMark_short')
    os.makedirs(plot_dir, exist_ok=True)
    goo = torch.squeeze(torch.cat((torch.from_numpy(X_train).float(), modelRINN_f0_mark_nonMark.Dyn_net.y_aug), dim=2).detach())
    mr = modelRINN_f0_mark_nonMark.Dyn_net.get_NonMarkClosure(goo).detach().numpy()
    f0_mark_train = modelRINN_f0_mark_nonMark.Dyn_net.get_mech(goo).detach().numpy()
    mr_true = np.squeeze(Grad_t) - f0_mark_train[:,:ode.K]
    for ind in range(ode.K):
        fig_path = os.path.join(plot_dir,'state{}'.format(ind))
        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))
        plt.plot(X_train[0,n_burnin:n_burnin+n_t1,ind], mr_true[n_burnin:n_burnin+n_t1,ind], linewidth=5, label='Residual derivative trajectory (T=1)')
        plt.plot(X_train[0,n_burnin:n_burnin+n_t1,ind], mr[n_burnin:n_burnin+n_t1,ind], linewidth=5, label='Learned derivative trajectory (T=1)')
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    plot_dir = os.path.join(params['path'], 'L96_residuals_nonMark_long')
    os.makedirs(plot_dir, exist_ok=True)
    goo = torch.squeeze(torch.cat((torch.from_numpy(X_train).float(), modelRINN_f0_mark_nonMark.Dyn_net.y_aug), dim=2).detach())
    mr = modelRINN_f0_mark_nonMark.Dyn_net.get_NonMarkClosure(goo).detach().numpy()
    f0_mark_train = modelRINN_f0_mark_nonMark.Dyn_net.get_mech(goo).detach().numpy()
    mr_true = np.squeeze(Grad_t) - f0_mark_train[:,:ode.K]
    for ind in range(ode.K):
        fig_path = os.path.join(plot_dir,'state{}'.format(ind))
        fig, axs = plt.subplots(ncols=1, figsize=(10, 10))
        plt.plot(X_train[0,n_burnin:,ind], mr_true[n_burnin:,ind], linewidth=1, label='Residual derivative trajectory (T=1)')
        plt.plot(X_train[0,n_burnin:,ind], mr[n_burnin:,ind], linewidth=1, label='Learned derivative trajectory (T=1)')
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    # fig_path = os.path.join(params['path'], 'L96_residuals_nonMark')
    # fig, axs = plt.subplots(ncols=d_obs, figsize=(200, 10))
    # for ind in range(d_obs):
    #     axs[ind].plot(goo[1000:,0], mr[1000:,0], label='non-Markovian example')
    # plt.savefig(fig_path)
    # plt.close()



    # plot the structure of the learnt markovian term
    # fig_path = os.path.join(params['path'], 'L96_residuals_Mark_nonMark')
    # plt.scatter(X_train.reshape(-1), (Grad_t - f0_train).reshape(-1))
    # m = np.squeeze(model_f0_mark_nonMark.f_nn_Mark(torch.FloatTensor(x_grid[:,None])).data.numpy())
    # plt.scatter(x_grid, m)
    # plt.savefig(fig_path)
    # plt.close()

    # generate test predictions by continuing the training set
    y0 = torch.cat((torch.from_numpy(X_train).float()[-1,-1:,:], modelRINN_f0_mark_nonMark.Dyn_net.y_aug[-1,-1:,:]), dim=1)
    # pred_f0_mark_nonMark = [y0]
    # for i in range(N_test_long):
    #     pred_f0_mark_nonMark.append(modelRINN_f0_mark_nonMark(pred_f0_mark_nonMark[-1],params['dt_integration'])[0])

    # bp()
    # integrate with solve_ivp
    settings= {'method': 'RK45'}
    sol_nonMark = my_solve_ivp( y0.data.numpy().reshape(-1), lambda t, y: model_f0_mark_nonMark.rhs(y,t), t_eval, t_span, settings)
    # sol_nonMark2 = my_solve_ivp( np.squeeze(y0.data.numpy()), lambda t, y: model_f0_mark_nonMark.rhs(y,t), t_eval, t_span, settings)
    # sol_nonMark3 = my_solve_ivp( np.squeeze(y0.data.numpy()), lambda t, y: model_f0_mark_nonMark.rhs(y,t), t_eval, t_span, settings)


    ###### plot predictions vs Test
    # remember, these are currently continuations of the train trajectory (hence warmup is "done")
    # fig_path = os.path.join(params['path'], 'trajectory_forecast_test')
    plot_dir = os.path.join(params['path'], 'trajectory_forecast_test')
    os.makedirs(plot_dir, exist_ok=True)
    for ind in range(d_obs):
        fig_path = os.path.join(plot_dir, 'state{}'.format(ind))
        fig, axs = plt.subplots(ncols=1, figsize=(20, 10))
        axs.plot(X_test[:N_test_short,ind], label='True')
        axs.plot(sol_f0[:N_test_short,ind], label='f0')
        axs.plot(sol_Mark[:N_test_short,ind], label='f0 + Mark')
        # axs.plot(sol_nonMark[:N_test_short,ind], label='f0 + Mark + NonMark')
        axs.plot(sol_nonMark[:,ind], label='f0 + Mark + NonMark')
        # axs.plot(torch.stack(pred_f0_mark_nonMark)[:N_test_short,0,ind].detach(), label='f0 + Mark + NonMark EXPLICIT')
        axs.legend()
        plt.savefig(fig_path)
        plt.close()

    fig_path = os.path.join(params['path'], 'hidden_dynamics_nonMark_testing_fit')
    fig, axs = plt.subplots(ncols=1, figsize=(20, 10))
    for ind in range(d_obs, sol_nonMark.shape[1]):
        axs.plot(sol_nonMark[:,ind], label='State {}'.format(ind-d_obs))
    axs.legend()
    plt.savefig(fig_path)
    axs.set_yscale('log')
    plt.savefig(fig_path+'_log')
    plt.close()

    ####### plot KDE statistics
    df = pd.DataFrame({'True': X_test[:N_test_long].reshape(-1),
                       'f0': sol_f0[:N_test_long,:d_obs].reshape(-1),
                       'f0 + Mark': sol_Mark[:N_test_long,:d_obs].reshape(-1),
                       # 'f0 + Mark + NonMark EXPLICIT': torch.stack(pred_f0_mark_nonMark)[:N_test_long,:,:d_obs].data.numpy().reshape(-1),
                       'f0 + Mark + NonMark': sol_nonMark[:N_test_long,:d_obs].reshape(-1)})
    sns.kdeplot(data=df)
    plt.savefig(fig_path_kde)
    plt.close()


if __name__ == '__main__':
    # eps_exp = int(sys.argv[1]) # time or noise
    # elist = [-7,-5,-3,-2,-1]
    # main(eps_exp=-1, N_lat=4)
    elist = [-1,-3,-5]
    for eps_exp in elist:
        for N_lat in [2,4,8]:
            try:
                main(eps_exp=eps_exp, N_lat=N_lat)
            except:
                pass
