import os
import numpy as np
import pickle
from pydoc import locate
import torch
from tqdm import tqdm

from computation_utils import computeErrors
from plotting_utils import *
from odelibrary import my_solve_ivp

from pdb import set_trace as bp

def get_Psi_ode(dt, rhs, integrator, t0=0):
	t_span = [t0, t0+dt]
	t_eval = np.array([t0+dt])
	settings = {}
	settings['dt'] = dt
	settings['method'] = integrator
	return lambda ic0: my_solve_ivp(ic=ic0, f_rhs=lambda t, y: rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)

class Psi(object):
	def __init__(self, dt, dynamics_rhs='L63', integrator='RK45', t0=0):
		self.settings = {'dt': dt, 'method': integrator}
		self.t_span = [t0, t0+dt]
		self.t_eval = np.array([t0+dt])
		ODE = locate('odelibrary.{}'.format(dynamics_rhs))
		self.ode = ODE()

	def step_wrap(self, ic):
		foo =  my_solve_ivp(ic=ic, f_rhs=lambda t, y: self.ode.rhs(y, t), t_eval=self.t_eval, t_span=self.t_span, settings=self.settings)
		return foo

class ENKF(object):
	def __init__(self,
				Psi,
				H,
				y_obs,
				dt,
				t0=0,
				v0_mean=None,
				v0_cov=None,
				output_dir='default_output_EnKF',
				N_particles=100,
				obs_noise_sd_assumed_enkf=0.1,
				obs_noise_sd_assumed_3dvar=0.1,
				obs_noise_sd_true=0,
				state_noise_sd=0,
				s_perturb_obs=True,
				rng_seed=0):

		np.random.seed(rng_seed)


		self.y_obs = y_obs # define the data

		self.N_filter = y_obs.shape[0]
		self.times_filter = np.arange(self.N_filter)

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.N_particles = N_particles
		self.H = H # linear observation operator for assimilation system
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T_filter = t0 + self.dt*self.N_filter
		self.obs_noise_sd_assumed_enkf = obs_noise_sd_assumed_enkf
		self.obs_noise_sd_assumed_3dvar = obs_noise_sd_assumed_3dvar
		self.obs_noise_sd_true = obs_noise_sd_true
		self.state_noise_sd = state_noise_sd
		self.s_perturb_obs = s_perturb_obs

		self.K_3dvar = self.H.T / (1+self.obs_noise_sd_assumed_3dvar)

		self.Psi_approx = Psi

		dim_x_approx = v0_mean.shape[0]
		dim_y = self.H.shape[0]
		self.obs_dim = dim_y
		self.hidden_dim = dim_x_approx - dim_y
		self.obs_noise_mean = np.zeros(dim_y)
		self.Gamma = (obs_noise_sd_assumed_enkf**2) * np.eye(dim_y) # obs_noise_cov
		self.Gamma_true = (obs_noise_sd_true**2) * np.eye(dim_y) # obs_noise_cov
		# self.y_obs = (self.H_true @ self.x_true.T).T + np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma_true, size=self.N_filter)

		# set up DA arrays
		# means
		self.x_pred_mean = np.zeros( (self.N_filter, dim_x_approx))
		self.y_pred_mean = np.zeros( (self.N_filter, dim_y))
		self.x_assim_mean = np.zeros_like(self.x_pred_mean)
		self.x_adhoc = np.zeros_like(self.x_pred_mean)
		self.x_assim_3dvar = np.zeros_like(self.x_pred_mean)
		self.x_pred_3dvar = np.zeros_like(self.x_pred_mean)
		self.y_pred_3dvar = np.zeros_like(self.y_pred_mean)

		# particles
		self.x_pred_particles = np.zeros( (self.N_filter, self.N_particles, dim_x_approx) )
		self.y_pred_particles = np.zeros( (self.N_filter, self.N_particles, dim_y) )
		self.x_assim_particles = np.zeros( (self.N_filter, self.N_particles, dim_x_approx) )

		#  error-collection arrays
		self.x_assim_error_mean = np.zeros_like(self.x_pred_mean)
		self.x_pred_mean_error = np.zeros_like(self.x_pred_mean)
		self.y_pred_mean_error = np.zeros_like(self.y_pred_mean)
		self.x_assim_error_particles = np.zeros( (self.N_filter, self.N_particles, dim_x_approx) )
		self.y_pred_error_particles = np.zeros( (self.N_filter, self.N_particles, dim_y) )
		self.x_adhoc_error = np.zeros_like(self.x_pred_mean)
		self.y_adhoc_error = np.zeros_like(self.y_pred_mean)
		self.x_assim_3dvar_error = np.zeros_like(self.x_pred_mean)
		self.y_assim_3dvar_error = np.zeros_like(self.y_pred_mean)
		self.x_pred_3dvar_error = np.zeros_like(self.x_pred_mean)
		self.y_pred_3dvar_error = np.zeros_like(self.y_pred_mean)

		# cov
		self.x_pred_cov = np.zeros((self.N_filter, dim_x_approx, dim_x_approx))

		# set up useful DA matrices
		self.Ix = np.eye(dim_x_approx)
		self.K_vec = np.zeros( (self.N_filter, dim_x_approx, dim_y) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)

		# choose ic for DA
		# x_ic_cov = (x_ic_sd**2) * np.eye(dim_x)
		# if x_ic_mean is None:
		# 	x_ic_mean = np.zeros(dim_x)

		v0 = np.random.multivariate_normal(mean=v0_mean, cov=v0_cov, size=self.N_particles)
		self.x_assim_particles[0] = np.copy(v0)
		self.x_pred_particles[0] = np.copy(v0)

		self.x_pred_mean[0] = np.mean(v0, axis=0)
		self.y_pred_mean[0] = self.H @ self.x_pred_mean[0]

	def roll_forward(self, ic, N, Psi_step):
		vec = np.zeros((N, ic.shape[0]))
		vec[0] = ic
		for n in range(1,N):
			vec[n] = Psi_step(vec[n-1])
		return vec

	def set_data(self, ic):
		foo = self.roll_forward(ic=ic, N=self.N_burnin + self.N_filter, Psi_step=self.Psi_true.step_wrap)
		self.x_true = foo[self.N_burnin:]

	def predict(self, ic):
		ic = torch.from_numpy(ic.astype(np.float32)[None,:])
		return np.squeeze(self.Psi_approx(ic, self.dt)[0].cpu().data.numpy())

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def update_3dvar(self, x_pred, y_obs):
		return (self.Ix - self.K_3dvar @ self.H) @ x_pred + (self.K_3dvar @ y_obs)

	def test(self):
		ic_dict = {
					'Ad hoc': self.x_adhoc[-1],
					'EnKF': self.x_assim_mean[-1],
					'3DVAR': self.x_assim_3dvar[-1],
					'3DVAR pred': self.x_pred_3dvar[-1],
					'True': self.x_true[-1]
					}
		# roll forward the predictions based on different ics
		traj_dict = [{} for _ in range(self.obs_dim)]
		for key in ic_dict:
			if key=='True':
				Psi_step = self.Psi_true
				H = self.H_true
			else:
				Psi_step = self.Psi_approx
				H = self.H
			foo_pred = self.roll_forward(ic=ic_dict[key], N=self.N_test, Psi_step=Psi_step)
			y_pred = (H @ foo_pred.T).T

			for i in range(self.obs_dim):
				traj_dict[i][key] = y_pred[:,i]

		for i in range(self.obs_dim):
			fig_path = os.path.join(self.output_dir, 'obs_test_predictions_dim{}'.format(i))
			plot_trajectories(times=self.times_test, traj_dict=traj_dict[i], fig_path=fig_path)

	def filter(self):
		# initialize adhoc---it is already all zeros, which will be used for hidden state
		# initialize 3dvar
		self.x_assim_3dvar[0,:self.obs_dim] = self.y_obs[0]

		# DA @ c=0, t=0 has been initialized already
		for c in tqdm(range(1, self.N_filter)):
			## predict
			self.t_pred += self.dt

			## run ad-hoc method
			# run prediction/update
			ic_adhoc = np.hstack((self.y_obs[c-1], self.x_adhoc[c-1,self.obs_dim:]))
			self.x_adhoc[c] = self.predict(ic=ic_adhoc)
			# compute errors
			try:
				self.x_adhoc_error[c] = self.x_true[c] - self.x_adhoc[c]
			except:
				pass
			self.y_adhoc_error[c] = self.H @self.x_adhoc_error[c]

			## run 3dvar method
			# 3dvar forecasts
			self.x_pred_3dvar[c] = self.predict(ic=self.x_assim_3dvar[c-1])
			self.y_pred_3dvar[c] = self.H @ self.x_pred_3dvar[c]
			# 3dvar updates
			self.x_assim_3dvar[c] = self.update_3dvar(x_pred=self.x_pred_3dvar[c], y_obs=self.y_obs[c])
			# compute errors
			try:
				self.x_pred_3dvar_error[c] = self.x_true[c] - self.x_pred_3dvar[c]
				self.x_assim_3dvar_error[c] = self.x_true[c] - self.x_assim_3dvar[c]
			except:
				pass
			self.y_pred_3dvar_error[c] = self.H @ self.x_pred_3dvar_error[c]
			self.y_assim_3dvar_error[c] = self.H @ self.x_assim_3dvar_error[c]

			# if not np.array_equal(self.x_pred_3dvar[c], self.x_adhoc[c]):
			# 	pdb.set_trace()

			## run EnKF method
			# compute and store ensemble forecasts
			for n in range(self.N_particles):
				self.x_pred_particles[c,n] = self.predict(ic=self.x_assim_particles[c-1,n])
				self.y_pred_particles[c,n] = self.H @ self.x_pred_particles[c,n]
			# compute and store ensemble means
			self.x_pred_mean[c] = np.mean(self.x_pred_particles[c], axis=0)
			self.y_pred_mean[c] = self.H @ self.x_pred_mean[c]

			# track assimilation errors for post-analysis
			# EnKF
			try:
				self.x_pred_mean_error[c] = self.x_true[c] - self.x_pred_mean[c]
			except:
				pass
			self.y_pred_mean_error[c] = self.H @ self.x_pred_mean_error[c]

			# compute and store ensemble covariance
			C_hat = np.cov(self.x_pred_particles[c], rowvar=False)
			self.x_pred_cov[c] = C_hat

			## compute gains for analysis step
			S = self.H @ C_hat @ self.H.T + self.Gamma
			self.K = C_hat @ self.H.T @ np.linalg.inv(S)
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c], axis=0)

			## assimilate
			self.t_assim += self.dt
			for n in range(self.N_particles):
				# optionally perturb the observation
				y_obs_n = self.y_obs[c] + self.s_perturb_obs * np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma)

				# prediction error for the ensemble member
				self.y_pred_error_particles[c,n] = self.y_obs[c] - self.y_pred_particles[c,n]

				# update particle
				self.x_assim_particles[c,n] = self.update(x_pred=self.x_pred_particles[c,n], y_obs=y_obs_n)

				# track assimilation errors for post-analysis
				try:
					self.x_assim_error_particles[c,n] = self.x_true[c] - self.x_assim_particles[c,n]
				except:
					pass

			# compute and store ensemble means
			self.x_assim_mean[c] = np.mean(self.x_assim_particles[c], axis=0)

			# track assimilation errors for post-analysis
			try:
				self.x_assim_error_mean[c] = self.x_true[c] - self.x_assim_mean[c]
			except:
				pass

		### compute evaluation statistics

		# Observed state evaluation
		fig_path = os.path.join(self.output_dir, 'assimilation_errors_obs')
		obs_eval_dict_3dvar_pred = computeErrors(target=self.y_obs, prediction=self.x_pred_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_3dvar = computeErrors(target=self.y_obs, prediction=self.x_assim_3dvar[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_enkf = computeErrors(target=self.y_obs, prediction=self.x_assim_mean[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		obs_eval_dict_adhoc = computeErrors(target=self.y_obs, prediction=self.x_adhoc[:,:self.obs_dim], dt=self.dt, thresh=0.05)
		errors = {'Ad hoc': obs_eval_dict_adhoc['mse'], 'EnKF':obs_eval_dict_enkf['mse'], '3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']   }
		plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

		# plot observed state synchronization (1st state only)
		fig_path = os.path.join(self.output_dir, 'obs_synchronization')
		traj_dict = {'Ad hoc': self.x_adhoc[:,0],
					'EnKF':self.x_assim_mean[:,0],
					'3DVAR':self.x_assim_3dvar[:,0],
					'3DVAR pred':self.x_pred_3dvar[:,0],
					'True':self.y_obs
					}
		plot_trajectories(times=self.times_filter, traj_dict=traj_dict, fig_path=fig_path)

		# Hidden/full state evaluation
		try:
			fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
			eval_dict_3dvar_pred = computeErrors(target=self.x_true, prediction=self.x_pred_3dvar, dt=self.dt, thresh=0.05)
			eval_dict_3dvar = computeErrors(target=self.x_true, prediction=self.x_assim_3dvar, dt=self.dt, thresh=0.05)
			eval_dict_enkf = computeErrors(target=self.x_true, prediction=self.x_assim_mean, dt=self.dt, thresh=0.05)
			eval_dict_adhoc = computeErrors(target=self.x_true, prediction=self.x_adhoc, dt=self.dt, thresh=0.05)
			errors = {'Ad hoc': eval_dict_adhoc['mse'], 'EnKF':eval_dict_enkf['mse'] ,'3DVAR':eval_dict_3dvar['mse'], '3DVAR pred':eval_dict_3dvar_pred['mse'] }
			plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			fig_path = os.path.join(self.output_dir, 'assimilation_errors_hidden')
			obs_eval_dict_3dvar_pred = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_pred_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_3dvar = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_assim_3dvar[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_enkf = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_assim_mean[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			obs_eval_dict_adhoc = computeErrors(target=self.x_true[:,self.obs_dim:], prediction=self.x_adhoc[:,self.obs_dim:], dt=self.dt, thresh=0.05)
			errors = {'Ad hoc': obs_eval_dict_adhoc['mse'], 'EnKF':obs_eval_dict_enkf['mse'], '3DVAR':obs_eval_dict_3dvar['mse'], '3DVAR pred':obs_eval_dict_3dvar_pred['mse']  }
			plot_assimilation_errors(times=self.times_filter, error_dict=errors, eps=self.obs_noise_sd_true, fig_path=fig_path)

			# plot hidden state synchronization (1st state only)
			fig_path = os.path.join(self.output_dir, 'hidden_synchronization')
			traj_dict = {'Ad hoc': self.x_adhoc[:,self.obs_dim+1],
						'EnKF': self.x_assim_mean[:,self.obs_dim+1],
						'3DVAR': self.x_assim_3dvar[:,self.obs_dim+1],
						'3DVAR pred': self.x_pred_3dvar[:,self.obs_dim+1],
						'True':self.x_true[:,self.obs_dim+1]
						}
			plot_trajectories(times=self.times_filter, traj_dict=traj_dict, fig_path=fig_path)
		except:
			print('True and Approximate Psi dimensions do not match; cannot evaluate hidden state assimilation.')
