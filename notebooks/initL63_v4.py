#!/usr/bin/env python
# coding: utf-8

import sys, os
import json
from time import time
sys.path.append("../code/modules")
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import PCA
import torch
from dynamical_models   import *
# from generate_data      import *
from NbedDyn            import *
from stat_functions     import *
from computation_utils import computeTestErrors
from plotting_utils import new_summary
from DA import ENKF
from odelibrary import L63, my_solve_ivp
import pandas as pd

from tqdm import tqdm

from pdb import set_trace as bp

def file_to_dict(fname):
	with open(fname) as f:
		my_dict = json.load(f)
	return my_dict


def generate_data(rng_seed, t_transient, t_data, delta_t, solver_type='default'):
	# load solver dict
	solver_dict='../Config/solver_settings.json'
	foo = file_to_dict(solver_dict)
	solver_settings = foo[solver_type]

	ode = L63()
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
	return x

def main(output_path, t_warmup=1, obs_noise_sd_true=1, traindata_ind=1, testdata_ind=1, trainseed=0, da_seed=0,
		thresh_list=np.array([0.4])):

	traindata_ind = int(traindata_ind)
	testdata_ind = int(testdata_ind)
	trainseed = int(trainseed)
	da_seed = int(da_seed)

	initial_cond_idx = 400

	# run the data generation
	class GD:
		model = 'Lorenz_63'
		class parameters:
			sigma = 10.0
			rho = 28.0
			beta = 8.0/3
		dt_integration = 0.01 # integration time
		dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
		dt_obs = 8# number of integration times between consecutive observations (for yo)
		var_obs = np.array([0,1]) # indices of the observed variables
		nb_loop_train = 110.01 # size of the catalog
		nb_loop_test = 100 # size of the true state and noisy observations
		sigma2_catalog = 0.0 # variance of the model error to generate the catalog
		sigma2_obs = 0.0 # variance of the observation error to generate observation
		obs_state_norm = 63 # approximate, from np.mean(X_train_chaos[:,0]**2)

	# catalog, xt, yo = generate_data(GD, rng_seed=traindata_ind)
	# X_train_chaos = xt.values
	X_train_chaos = generate_data(rng_seed=traindata_ind, t_transient=100, t_data=100, delta_t=GD.dt_integration)

	# catalog, xt, yo = generate_data(GD, rng_seed=10*testdata_ind)
	# X_test = xt.values[:int(8/GD.dt_integration)]
	X_test = generate_data(rng_seed=10*testdata_ind, t_transient=100, t_data=15, delta_t=GD.dt_integration)

	length_prior = int(t_warmup/GD.dt_integration)


	# Define noisy data
	np.random.seed(testdata_ind)
	obs = np.copy(X_test[initial_cond_idx-length_prior:initial_cond_idx,0])
	obs += obs_noise_sd_true*np.random.normal(0,1,obs.shape[0])
	obs = obs[:,None]
	np.random.seed()



	X_train    = X_train_chaos[:,:1]#[:-1,:1]
	Grad_t     = np.gradient(X_train[:,0]).reshape(X_train.shape[0],1)/GD.dt_integration
	Batch_size = X_train.shape[0]
	nb_batch   = int(X_train.shape[0]/Batch_size)
	X_train    = X_train.reshape(nb_batch,Batch_size,1)
	Grad_t     = Grad_t.reshape(nb_batch,Batch_size,1)



	init_dict_opt = {} # collection of initial conditions to compare
	init_dict_da = {}
	plot_dict = {}

	N_lat = 2
	params = {}
	params['seed']               = trainseed
	params['transition_layers']  = 1
	params['bi_linear_layers']   = N_lat+1
	params['dim_hidden_linear']  = N_lat+1
	params['dim_input']          = 1
	params['dim_latent']         = N_lat
	params['dim_observations']   = 1
	params['dim_Embedding']      = N_lat+1
	params['ntrain']             = [30000,1000]
	# params['ntrain']             = [3000,100] # NOTE I MADE THIS SHORTER for ease
	params['dt_integration']     = 0.01
	params['pretrained']         = False
	params['nb_batch']           = nb_batch
	params['Batch_size']         = Batch_size
	params['get_latent_train']   = False
	params['path']               = os.path.join(output_path,'trainData{}_trainSeed{}/'.format(traindata_ind, trainseed))
	test_path = os.path.join(params['path'],'noise{}_twarmup{}'.format(obs_noise_sd_true, t_warmup), 'testData{}'.format(testdata_ind))
	os.makedirs(params['path'], exist_ok=True)
	os.makedirs(test_path, exist_ok=True)
	params['file_name']    = 'NbedDyn'
	model, modelRINN = get_NbedDyn_model(params)
	model, modelRINN, aug_inp_data = train_NbedDyn_model_L63(params,model,modelRINN,X_train,Grad_t)
	pred = [torch.cat((torch.from_numpy(X_train).float()[-1,-1:,:], modelRINN.Dyn_net.y_aug[-1,-1:,:]), dim=1)]

	print('Generating final training-set predictions')
	for i in range(10000):
		pred.append(modelRINN(pred[-1],params['dt_integration'])[0])

	forecasting_err=np.zeros(1000)
	for i in range(1,100):
		forecasting_err[i-1]=RMSE(X_test[i-1,0],torch.stack(pred).data.numpy()[i,0,0])

	print('prediction error at t0 + dt : '  ,forecasting_err[0])
	print('prediction error at t0 + 4dt : ' ,forecasting_err[3])

	# SKIPPING LYAPUNOV STUFF BECAUSE IT IS SLOW AND NOT RELEVANT TO INITIALIZATION
#         l_exp, l_dim = Compute_Lyapunov_spectrum(modelRINN, np.concatenate((X_train[-1,-1,:], modelRINN.Dyn_net.y_aug.detach().numpy()[-1,-1,:]), axis=0), 10000, 0.1, 0.01, True)
#         lyap_series =  compute_largest_Lyapunov(modelRINN, np.concatenate((X_train[-1,-1,:], modelRINN.Dyn_net.y_aug.detach().numpy()[-1,-1,:]), axis=0),0.01,0.1,10000, True)[0]

#         print(l_exp)
#         print(lyap_series[-1])
#         print(l_dim)

	output_results = {}
	output_results['description']    = 'nbedDyn_model_output_dim'+str(params['dim_observations'])+'_seed_'+str(params['seed'])
	output_results['pred']           = torch.stack(pred).data.numpy()
#         output_results['lyap_spect']     = l_exp
#         output_results['lyap_dim']       = l_dim
#         output_results['largest_lyap']   = lyap_series
	output_results['forecast_error'] = forecasting_err
	# write python dict to a file
	output = open(os.path.join(params['path'], params['file_name'] + '.pkl'), 'wb')
	pickle.dump(output_results, output)
	output.close()


	# plot dynamics of learned model---assess the variance of each state for good EnKF initialization
	fig_path = os.path.join(params['path'],'trained_dynamics')
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
	pred_vec = np.squeeze(torch.stack(pred).data.numpy())
	ax.plot(pred_vec[:2000,:])
	plt.savefig(fig_path)
	plt.close()


	model_cov = np.diag(np.var(pred_vec,axis=0))


	# # Simple method : find analogs for the latent states in the training set:


	def get_init(test_vars,length_prior,dt,dim_aug):
		x = torch.from_numpy(X_train).float()
		aug_inp = torch.cat((x[0,:,:], modelRINN.Dyn_net.y_aug[0,:,:]), dim=1)
		pred, grad, inp, aug_inp = modelRINN(aug_inp,dt)
		pred, aug = model_Multi_RINN_simple(aug_inp, 0.0, length_prior, dt)
		loss_init=[]
		for i in range(aug_inp.shape[0]):
			loss_init.append(((pred[1:,i,:1][torch.where(~torch.isnan(test_vars))]-test_vars[torch.where(~torch.isnan(test_vars))])**2).mean())
		min_idx = np.where(torch.stack(loss_init).data.numpy()==torch.stack(loss_init).data.numpy().min())
		inp_init = aug_inp.detach().data.numpy()[min_idx[0][0]].reshape(1,dim_aug)
		inp_init = (torch.from_numpy(inp_init).float())
		return inp_init
	class Multi_INT_net(torch.nn.Module):
			def __init__(self, params):
				super(Multi_INT_net, self).__init__()
	#            self.add_module('Dyn_net',FC_net(params))
				self.Int_net = modelRINN
			def forward(self, inp, t0, nb, dt):
				"""
				In the forward function we accept a Tensor of input data and we must return
				a Tensor of output data. We can use Modules defined in the constructor as
				well as arbitrary operators on Tensors.
				"""
	#            dt = Variable(torch.from_numpy(np.reshape(dt,(1,1))).float())
	#            x = Variable(3*torch.ones(1, 1), requires_grad=True)

				#grad, aug_inp = self.Dyn_net(inp,dt)
				#pred = aug_inp +dt*grad
				pred = [inp]
				aug  = []
				for i in range(nb):
					predic, k1, inp, aug_inp = self.Int_net(pred[-1], dt)
					pred.append(predic)
					aug.append(aug_inp)
				return torch.stack(pred), torch.stack(aug)
	model_Multi_RINN_simple = Multi_INT_net(params)
	criterion = torch.nn.MSELoss()


	# In[21]:

	print('Doing KNN-initialization using training set...')
	test_vars = (torch.from_numpy(np.reshape(obs,(length_prior,1))).float())
	inp_init_knn = get_init(test_vars,length_prior,0.01,params['dim_Embedding'])
	init_dict_opt['knn'] = inp_init_knn

	print('Simulating KNN-based prediction...')
	y_pred2=np.zeros((2000+length_prior+1,params['dim_Embedding']))
	tmp = inp_init_knn
	y_pred2[0,:] = tmp.cpu().data.numpy()
	for k in range(1,2000+length_prior+1):
		y_pred2[k,:] = modelRINN(tmp,0.01)[0].cpu().data.numpy()
		tmp = (torch.from_numpy(np.reshape(y_pred2[k,:],(1,params['dim_Embedding']))).float())

	# define DA scheme for identifying inits from noisy data
	def get_initial_condition_DA(model, time_series, dt, obs_noise_sd_true, train_init=None, model_cov=None, rng_seed=da_seed):

		if train_init is None:
			v0_mean = np.zeros(3)
		else:
			v0_mean = np.copy(train_init)

		if model_cov is None:
			model_cov = 10*np.eye(3)

		H = np.zeros((1, 3))
		for ob in range(1):
			H[ob, ob] = 1

		da_params = {}
		da_params['rng_seed'] = rng_seed
		da_params['Psi'] = model
		da_params['H'] = H
		da_params['y_obs'] = time_series
		da_params['dt'] = dt
		da_params['t0']= 0
		da_params['v0_mean'] = v0_mean
		da_params['v0_cov'] = 100*model_cov
		da_params['output_dir'] = 'DAinit'
		da_params['N_particles'] = 100
		da_params['obs_noise_sd_true'] = obs_noise_sd_true
		da_params['obs_noise_sd_assumed_enkf'] = 1*obs_noise_sd_true
		da_params['obs_noise_sd_assumed_3dvar'] = 1*obs_noise_sd_true
		da_params['state_noise_sd'] = 0
		da_params['s_perturb_obs'] = True # option to further perturb measurements in the DA scheme

		da = ENKF(**da_params)
		da.filter()

		init_dict = {
					'ad hoc': da.x_adhoc,
					'EnKF': da.x_assim_mean,
					'3DVAR': da.x_assim_3dvar}
	#                 '3DVAR pred': da.x_pred_3dvar}
		return init_dict

	print('Running DA-based initialization inference...')
	all_dict_da = get_initial_condition_DA(modelRINN, obs, 0.01, obs_noise_sd_true, model_cov=model_cov, rng_seed=19*da_seed)

	## Plot comparison of DA-based initial conditions in a test setting by iterating them forwards

	# set color_key
	colors = {'EnKF': 'orange', 'ad hoc': 'blue',
			  'True': 'black', '3DVAR': 'green',
			  'knn': 'purple',
			  'torch.opt': 'magenta'}
	# set dict
	init_dict_da = {key: all_dict_da[key][-1,None] for key in all_dict_da}

	fig_path = os.path.join(test_path, 'DA_inits')
	fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
	ax1.plot(X_test[initial_cond_idx-length_prior:,0], label='True', color='black')
	ax1.plot(obs, label='Observed', color='red') #400:500
	ntest = X_test.shape[0] - initial_cond_idx

	print('Simulating test-set predictions from DA-based initialization...')
	for key in init_dict_da:
		y_pred=np.zeros((ntest,params['dim_Embedding']))
		mse=np.zeros(ntest)
		mse_cum=np.zeros(ntest)
		mse_all=np.zeros(ntest)
		mse_all_cum=np.zeros(ntest)

		tmp = torch.from_numpy(init_dict_da[key].astype(np.float32))
		y_pred[0,:] = tmp.cpu().data.numpy()
		for k in range(1,ntest):
			y_pred[k,:] = modelRINN(tmp,0.01)[0].cpu().data.numpy()
			tmp = (torch.from_numpy(np.reshape(y_pred[k,:],(1,params['dim_Embedding']))).float())
			mse[k] = np.mean((y_pred[k,0] - X_test[initial_cond_idx+k,0])**2)
			mse_cum[k] = mse_cum[k-1] + mse[k]
			mse_all[k] = np.mean((y_pred[k] - X_test[initial_cond_idx+k])**2)
			mse_all_cum[k] = mse_all_cum[k-1] + mse_all[k]
		plot_dict[key] = {'mse_cum': mse_cum, 'mse_all_cum': mse_all_cum, 'x':length_prior + np.arange(ntest), 'y_pred':y_pred[:,0]}
		ax1.plot(length_prior + np.arange(ntest),y_pred[:,0], label=key, color=colors[key])
	#     ax1.plot(np.arange(ntest), y_pred3[:,0], label=key)
		ax2.plot(mse_cum, label=key, color=colors[key])



	ax1.legend()
	ax1.set_title('Dynamics')
	ax2.set_title('Error in observed-state prediction')
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	ax2.set_xlabel('time')
	ax1.set_xlabel('time')
	ax2.legend()
	plt.savefig(fig_path)
	plt.close()

	# define variational assimilation problem
	def get_initial_condition(model, time_series, train_series, dt,lr_init, err_tol = 1E-4, n_train = 10000, train_init = None, rng_seed=da_seed):
		torch.manual_seed(rng_seed)
		criterion = torch.nn.MSELoss()#reduction = 'sum')
		if train_init is None:
			min_idx = None
			inp_init = torch.rand(1,(train_series.shape[-1]+model.Int_net.Dyn_net.y_aug.shape[-1])).float()*0.0
			inp_init[:,:time_series.shape[-1]] = time_series.clone()[:1,:]
		else:
			inp_init = train_init
	#         print(time_series.shape)
	#         inp_init = get_init(time_series,time_series.shape[0],0.01,params['dim_Embedding'])

		init_cond_model = get_init_model(model,inp_init)
		optimizer = torch.optim.Adam(init_cond_model.parameters(), lr = lr_init)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, factor = 0.1, patience=205, verbose=True, min_lr = 0.0001)
		stop_cond = False
		count = 0
		while(stop_cond==False):
			# Forward pass: Compute predicted y by passing x to the model
			pred = init_cond_model(0.0,time_series.shape[0],dt)
			#pred1, grad, inp, aug_inp = modelRINN(test_vars[:1,:],dt, True, iterate = t)
			# Compute and print loss
			loss = torch.mean((pred[1:,0,:time_series.shape[-1]][torch.where(~torch.isnan(time_series))]- time_series[torch.where(~torch.isnan(time_series))])**2)
	#        criterion(pred[1:,0,:time_series.shape[-1]], time_series[:,:])
			print(count,loss)
			# Zero gradients, perform a backward pass, and update the weights.
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			scheduler.step(loss)
			count += 1
			if loss.detach().numpy()<err_tol or count>n_train:
				stop_cond = True
		return init_cond_model.estimate_init

	class get_init_model(torch.nn.Module):
			def __init__(self, model_Multi_RINN, inp_init):
				super(get_init_model, self).__init__()
				self.Multi_INT_net = model_Multi_RINN
				self.estimate_init = torch.nn.Parameter((inp_init.clone()))#torch.nn.Parameter(aug_inp[:1,:])
			def forward(self, t0, nb, dt):
				pred = self.Multi_INT_net(self.estimate_init, t0, nb, dt)[0]
				return pred


	# define a copy of the model for which we only optimize ICs.
	print('Running torch.opt-based initialization...')
	model_Multi_RINN = Multi_INT_net(params)
	for param in model_Multi_RINN.Int_net.parameters():
		param.requires_grad = False

	# run variational assimilation with pytorch optimization
	train_series = torch.from_numpy(X_train).float()
	# inp_init_opti = get_initial_condition(model_Multi_RINN, torch.from_numpy(obs).float(), train_series, 0.01,lr_init = 0.001, err_tol = 1E-1, n_train = 1000, train_init = inp_init_knn)
	inp_init_opti = get_initial_condition(model_Multi_RINN, torch.from_numpy(obs).float(), train_series, 0.01,lr_init = 0.001, err_tol = 1E-1, n_train = 100, train_init = inp_init_knn, rng_seed=21*da_seed)
	init_dict_opt['torch.opt'] = inp_init_opti
	# inp_init_knn, inp_init_opti = get_initial_condition(model_Multi_RINN, torch.from_numpy(observations[initial_cond_idx:initial_cond_idx+length_prior,:1]).float(), train_series, 0.01,lr_init = 0.001, err_tol = 1E-1, n_train = 10000,

	# add optimization and KNN methods to plot for comparison
	print('Simulating test-set predictions from torch.opt-based initialization...')
	for key in init_dict_opt:
		tmp = init_dict_opt[key].data
		for k in range(1,length_prior):
			tmp = modelRINN(tmp,0.01)[0]

		y_pred=np.zeros((ntest,params['dim_Embedding']))
		mse=np.zeros(ntest)
		mse_cum=np.zeros(ntest)
		mse_all=np.zeros(ntest)
		mse_all_cum=np.zeros(ntest)
		y_pred[0,:] = tmp.cpu().data.numpy()
		for k in range(1,ntest):
			y_pred[k,:] = modelRINN(tmp,0.01)[0].cpu().data.numpy()
			tmp = (torch.from_numpy(np.reshape(y_pred[k,:],(1,params['dim_Embedding']))).float())
			mse[k] = np.mean((y_pred[k,0] - X_test[initial_cond_idx+k,0])**2)
			mse_cum[k] = mse_cum[k-1] + mse[k]
			mse_all[k] = np.mean((y_pred[k] - X_test[initial_cond_idx+k])**2)
			mse_all_cum[k] = mse_all_cum[k-1] + mse_all[k]
		plot_dict[key] = {'mse_cum': mse_cum, 'mse_all_cum': mse_all_cum, 'x':length_prior + np.arange(ntest), 'y_pred':y_pred[:,0]}

	fig_path = os.path.join(test_path, 'all_inits')
	fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
	ax1.plot(X_test[initial_cond_idx-length_prior:,0], label='True', color='black')
	ax1.plot(obs, label='Observed', color='red') #400:500
	for key in init_dict_da.keys():
		foo = plot_dict[key]
		ax1.plot(foo['x'], foo['y_pred'], label=key, color=colors[key])
		ax2.plot(foo['mse_cum'], label=key, color=colors[key])

	for key in init_dict_opt.keys():
		foo = plot_dict[key]
		ax1.plot(foo['x'], foo['y_pred'], label=key, color=colors[key])
		ax2.plot(foo['mse_cum'], label=key, color=colors[key])


	ax1.legend()
	ax1.set_title('Dynamics')
	ax2.set_title('Error in observed-state prediction')
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	ax2.set_xlabel('time')
	ax1.set_xlabel('time')
	ax2.legend()
	plt.savefig(fig_path)
	plt.close()

	# evaluate performance on test-set
	print('Computing and saving test-set errors for all methods...')
	target = X_test[initial_cond_idx:,0].reshape(-1,1)
	eval_dict = {}
	for key in plot_dict:
		prediction = plot_dict[key]['y_pred'].reshape(-1,1)
		eval_dict[key] = computeTestErrors(target=target, prediction=prediction, dt=GD.dt_integration, thresh_norm=GD.obs_state_norm, thresh_list=thresh_list)
		nm_list = [q for q in eval_dict[key].keys() if q != 'mse']
		mystr = ['{}={}'.format(q, eval_dict[key][q]) for q in nm_list]
		mystr = ', '.join(mystr)
		print(key, mystr)
		# print(key, ':', 't_valid_005=', eval_dict[key]['t_valid_005'], 'mse_total=', eval_dict[key]['mse_total'])


	# save evaluation
	output = open(os.path.join(test_path, 'eval_data.pkl'), 'wb')
	pickle.dump(eval_dict, output)
	output.close()

	return eval_dict

def big_runner(runtype='time', output_nm='../ic_outputs_L63/default_output',
				ntrain = 3,
				ntest = 10,
				t_warmup_list=[1],
				noise_list=[1],
				thresh_list=np.array([0.005, 0.05, 0.2, 0.4])):

	lookup = {'time': {'x': 't_warmup', 'xlabel': 'Warmup Time'},
			'noise': {'x': 'obs_noise_sd_true', 'xlabel': 'Observation Noise SD'} }

	xlabel = lookup[runtype]['xlabel']
	xnm = lookup[runtype]['x']

	y_list =['t_valid_{}'.format(t) for t in thresh_list]

	def make_plots(df, xnm=xnm, xlabel=xlabel):
		for y in y_list:
			fig_path = output_nm+'_{}_{}'.format(runtype, y.replace('.','-'))
			new_summary(df, fig_path, hue='index', style='index', x=xnm, y=y,
						ylabel='Validity Time',
						xlabel=xlabel,
						title='Initialization Performance',
						estimator=np.mean,
						ci='sd',
						legloc='upper right')

		fig_path = output_nm+'_{}_mse'.format(runtype)
		new_summary(df, fig_path, hue='index', style='index', x=xnm, y='mse_total',
					ylabel='MSE',
					xlabel=xlabel,
					title='Initialization Performance',
					estimator=np.mean,
					ci='sd',
					legloc='upper right')

	# create / read job queue
	job_file = output_nm + '_{}_job.csv'.format(runtype)
	if os.path.isfile(job_file):
		job_df = pd.read_csv(job_file)
		# sort job
		job_df.sort_values(by=[lookup[runtype]['x'], 'traindata_ind', 'testdata_ind'], inplace=True)
	else:
		job_df = pd.DataFrame()

	job_list = []
	for traindata_ind in range(ntrain):
		for testdata_ind in range(ntest):
			for t_warmup in t_warmup_list:
				for obs_noise_sd_true in noise_list:
					maindict = {'traindata_ind': traindata_ind,
								'testdata_ind': testdata_ind,
								't_warmup': t_warmup,
								'obs_noise_sd_true': obs_noise_sd_true,
								'done': 0}
					job_list.append(maindict)
	# append all jobs to existing jobs
	job_df = job_df.append(pd.DataFrame(job_list))
	# remove completed jobs which present as duplicates with done=0 (use max)
	job_df = job_df.groupby(['traindata_ind','testdata_ind','t_warmup','obs_noise_sd_true'], as_index=False).agg('max')
	job_df.to_csv(job_file, index=False)

	job_inds = job_df.index[job_df['done']==0]

	# create / read job output file
	csv_name = output_nm+'_{}.csv'.format(runtype)
	try:
		df_time = pd.read_csv(csv_name)
		make_plots(df_time)
	except:
		df_time = pd.DataFrame()

	# now run jobs according to job_df
	for j in tqdm(job_inds):
		job = job_df.iloc[j]
		run_dict = job.drop(['done']).to_dict()

		# run!
		eval_dict = main(output_path=output_nm, thresh_list=thresh_list, **run_dict)

		# process output
		foo = pd.DataFrame(eval_dict).T
		foo.drop(columns=['mse'], inplace=True)
		cols = foo.columns[foo.dtypes.eq('object')]
		foo[cols] = foo[cols].apply(pd.to_numeric)
		foo.reset_index(inplace=True)
		foo['traindata_ind'] = run_dict['traindata_ind']
		foo['testdata_ind'] = run_dict['testdata_ind']
		foo['t_warmup'] = run_dict['t_warmup']
		foo['obs_noise_sd_true'] = run_dict['obs_noise_sd_true']

		# add output to results df
		df_time = df_time.append(foo)
		df_time.to_csv(csv_name, index=False)

		make_plots(df_time)

		# mark job done
		job_df.loc[j,'done'] = 1
		# save job
		job_df.to_csv(job_file, index=False)



if __name__ == '__main__':
	output_nm = '../ic_outputs_L63/experiment1_redo/'
	os.makedirs(output_nm, exist_ok=True)

	runtype = sys.argv[1] # time or noise
	if runtype=='noise':
		noise_list = [1e-3, 1e-2, 1e-1, 1, 2, 5]
		t_warmup_list = [1]
	elif runtype=='time':
		noise_list = [1]
		t_warmup_list = [3, 2, 1, 0.5, 0.1]

	big_runner(runtype=runtype, output_nm=output_nm,
				ntrain=6,
				ntest=20,
				t_warmup_list=t_warmup_list,
				noise_list=noise_list)
