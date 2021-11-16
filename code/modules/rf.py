import os
import numpy as np
from scipy.linalg import pinv2 as scipypinv2
import matplotlib
from matplotlib import rc, cm, colors
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
import torch

from pdb import set_trace as bp

class RF(object):
	def __init__(self,
					do_normalization=False,
					activation = torch.cos,
					Dx = 1,
					Dr = 20,
					fac_w = 2,
					fac_b = 2,
					lam_rf = 1e-5,
					fig_path= 'test4_output',
					seed=0):
		np.random.seed(seed)

		self.do_normalization = do_normalization
		self.fig_path = fig_path
		os.makedirs(self.fig_path, exist_ok=True)
		self.activation = activation
		self.fac_w = fac_w
		self.fac_b = fac_b
		self.Dx = Dx
		self.Dy = 1
		self.Dr = Dr
		self.lam_rf = lam_rf
		self.set_rf() # draw random features

	def set_rf(self):
		self.w_in = torch.FloatTensor(self.fac_w * np.random.randn(self.Dr, self.Dx))
		self.b_in = torch.FloatTensor(np.random.uniform(low=0, high=2*np.pi, size= (self.Dr, 1)))

		# self.w_in = np.random.uniform(low=-self.fac_w, high=self.fac_w, size= (self.Dr, self.Dx))
		# self.b_in = np.random.uniform(low=-self.fac_b, high=self.fac_b, size= (self.Dr, 1))

	def predict(self, x_input, make_plots=False, use_torch=True):
		self.normalize_data(x_input, save=False) # store and normalize data

		if use_torch:
			self.x_norm = torch.FloatTensor(self.x_norm)

		Phi = self.compute_Phi(self.x_norm)

		if use_torch:
			y_pred_scaled = self.C @ Phi
		else:
			y_pred_scaled = self.C.data.numpy() @ Phi.data.numpy()

		y_pred = self.descaleY(y_pred_scaled)

		# plot RFs
		if make_plots:
			if self.Dx==1:
				self.plot_rf_d1(self.descaleX(self.x_norm.T), self.Phi_lib(self.x_norm.T), nm='rf_functions')
			elif self.Dx==2:
				self.plot_rf_d2(f=self.phi_j, J=self.Dr, nm='rf_functions')
			else:
				print('Couldnt plot RF')

		return y_pred


	def fit(self, x_input, y_output):
		self.normalize_data(x_input, y_output, save=True) # store and normalize data

		### Compute libraries
		print('Computing Phi and Library matrices...')
		Phi = self.compute_Phi(self.x_norm).data.numpy()

		### Fit whole function with RF
		Nx = x_input.shape[1]
		lam = self.lam_rf * Nx / (self.Dy)
		C = self.y_norm @ Phi.T @ scipypinv2(Phi@Phi.T + lam * np.eye(self.Dr))

		#aggregate
		y_pred_scaled = C @ Phi
		self.y_fit = self.descaleY(y_pred_scaled)

		self.C = torch.FloatTensor(C)
		# self.plot_rf_seq((y_pred_scaled.T - self.y_norm.T)**2, nm='rfPred_sqerr_seq')
		# self.plot_rf_seq((y_pred_scaled.T - self.y_norm.T)/self.y_norm.T, nm='rfPred_relerr_seq')
		# self.plot_rf_seq(y_pred_scaled.T, nm='rfPred_seq')

	def compute_Phi(self, X):
		Phi = self.activation(self.w_in @ X + self.b_in)
		return Phi

	def normalize_data(self, x_input, y_output=None, save=True):
		self.x_norm = self.scaleX(x_input, save=save)
		if y_output is not None:
			self.y_norm = self.scaleY(y_output, save=save)

	def scaleX(self, x, save=False):
		if self.do_normalization:
			if save:
				self.x_mean = np.mean(x)
				self.x_std = np.std(x)
			return (x-self.x_mean) / self.x_std
		else:
			return x

	def descaleX(self, x):
		if self.do_normalization:
			return self.x_mean + (self.x_std * x)
		else:
			return x

	def scaleY(self, y, save=False):
		if self.do_normalization:
			if save:
				self.y_mean = np.mean(y)
				self.y_std = np.std(y)
			return (y-self.y_mean) / self.y_std
		else:
			return y

	def descaleY(self, y):
		if self.do_normalization:
			return self.y_mean + (self.y_std * y)
		else:
			return y

	def plot_rf_d1(self, x,  Phi, nm='RF_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		N, D = Phi.shape
		for d in range(D):
			ax.plot(x, Phi[:,d], label='f_{}'.format(d))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))
		plt.close()

	def plot_rf_seq(self, Phi, nm='RF_seq_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		N, D = Phi.shape
		for d in range(min(10,D)):
			ax.plot(Phi[:,d], label='f_{}'.format(d))
			plt.savefig(os.path.join(self.fig_path, nm))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))
		plt.close()

	def plot_rf_d2(self, f, J, nm='RF_functions'):
		for j in range(min(J,10)):
			Phi = f(j, self.XY_grid)
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
			ax.scatter(x=self.XY_grid[:,0], y=self.XY_grid[:,1], c=Phi)
			plt.savefig(os.path.join(self.fig_path, nm+str(j)))
			plt.close()



if __name__ == '__main__':
	rf = RF()
	rf.run()
