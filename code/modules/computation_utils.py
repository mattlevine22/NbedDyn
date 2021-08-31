#!/usr/bin/env python
import numpy as np
import pickle
import io
import os
from scipy.stats import gaussian_kde, entropy
from statsmodels.tsa.stattools import acf

import pdb

def mse(x, y):
	return np.mean( (x-y)**2 , axis=0)

def matt_xcorr(x, y):
	foo = np.correlate(x, y, mode='full')
	normalization = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
	xcorr = np.true_divide(foo,normalization)
	return xcorr

def linear_interp(x_vec, n_min, t, t0, dt):
	ind_mid = (t-t0) / dt
	ind_low = max(0, min( int(np.floor(ind_mid)), n_min) )
	ind_high = min(n_min, int(np.ceil(ind_mid)))
	v0 = x_vec[ind_low,:]
	v1 = x_vec[ind_high,:]
	tmid = ind_mid - ind_low
	return (1 - tmid) * v0 + tmid * v1


def replaceNaN(data):
	data[np.isnan(data)]=float('Inf')
	return data

def kde_scipy(x, x_grid, **kwargs):
	"""Kernel Density Estimation with Scipy"""
	# Note that scipy weights its bandwidth by the covariance of the
	# input data.  To make the results comparable to the other methods,
	# we divide the bandwidth by the sample standard deviation here.
	kde = gaussian_kde(x, **kwargs)
	return kde.evaluate(x_grid)

def kl4dummies(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=1000):
	# arrays are identical and KL-div is 0
	if np.array_equal(Xtrue, Xapprox):
		return 0
	# compute KL-divergence
	x_grid, Pk, Qk = kdegrid(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=gridsize)
	kl = entropy(Pk, Qk) # compute Dkl(P | Q)
	return kl

def kdegrid(Xtrue, Xapprox, kde_func=kde_scipy, gridsize=1000):
	zmin = min(min(Xtrue), min(Xapprox))
	zmax = max(max(Xtrue), max(Xapprox))
	x_grid = np.linspace(zmin, zmax, gridsize)
	Pk = kde_func(Xapprox.astype(np.float), x_grid) # P is approx dist
	Qk = kde_func(Xtrue.astype(np.float), x_grid) # Q is reference dist
	return x_grid, Pk, Qk

def computeErrors(target, prediction, dt, thresh):
	prediction = replaceNaN(prediction)

	# MEAN (over-space) SQUARE ERROR
	mse = np.mean((target-prediction)**2, axis=1)

	# TIME-AVG of MSE
	mse_total = np.mean(mse)

	# time convergence
	t_convergence = dt*getNumberOfBadPredictions(mse, thresh)

	eval_dict = {}
	for var_name in ['mse', 'mse_total', 't_convergence']:
		exec("eval_dict['{key}'] = {val}".format(key=var_name, val=var_name))
	return eval_dict


def getNumberOfBadPredictions(nerror, tresh=0.05):
	nerror_bool = nerror > tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n

def getNumberOfAccuratePredictions(nerror, tresh=0.05):
	nerror_bool = nerror < tresh
	n_max = np.shape(nerror)[0]
	n = 0
	while nerror_bool[n] == True:
		n += 1
		if n == n_max: break
	return n


class scaler(object):
	def __init__(self, tt, tt_derivative, component_wise=False):

		self.component_wise = component_wise
		self.tt = tt
		self.data_min = 0
		self.data_max = 0
		self.data_mean = 0
		self.data_std = 0

		# scale derivatives
		self.tt_derivative = tt_derivative
		self.derivative_min = 0
		self.derivative_max = 0
		self.derivative_mean = 0
		self.derivative_std = 0


	def scaleData(self, input_sequence, reuse=False):
		# data_mean = np.mean(train_input_sequence,0)
		# data_std = np.std(train_input_sequence,0)
		# train_input_sequence = (train_input_sequence-data_mean)/data_std
		if self.component_wise:
			axis = None
		else:
			axis = 0
		if not reuse:
			self.data_mean = np.mean(input_sequence,axis=axis)
			self.data_std = np.std(input_sequence,axis=axis)
			self.data_min = np.min(input_sequence,axis=axis)
			self.data_max = np.max(input_sequence,axis=axis)

			if self.tt in ["Standard2", "standard2"]:
				self.data_mean = np.mean(self.data_mean)
				self.data_std = np.std(self.data_std)
				self.data_min = np.min(self.data_min)
				self.data_max = np.max(self.data_max)
			if self.tt in ["Standard3", "standard3"]:
				self.data_std = np.std(self.data_std)
				self.data_min = np.min(self.data_min)
				self.data_max = np.max(self.data_max)

		if self.tt in ["Standard", "standard", "Standard2", "standard2", "Standard3", "standard3"]:
			input_sequence = (input_sequence-self.data_mean)/self.data_std
		elif self.tt == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence-self.data_min)/(self.data_max-self.data_min))
		elif self.tt == "meanOnly":
			input_sequence = np.array(input_sequence-self.data_mean)
		elif self.tt == "stdOnly":
			input_sequence = np.array(input_sequence/self.data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleData(self, input_sequence):
		if self.tt in ["Standard", "standard", "Standard2", "standard2", "Standard3", "standard3"]:
			input_sequence = input_sequence*self.data_std.T + self.data_mean
		elif self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.data_max - self.data_min) + self.data_min)
		elif self.tt == "meanOnly":
			input_sequence = np.array(input_sequence+self.data_mean)
		elif self.tt == "stdOnly":
			input_sequence = np.array(input_sequence*self.data_std.T)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def scaleXdot(self, input_sequence):
		if self.tt in ["Standard", "standard", "Standard2", "standard2", "Standard3", "standard3"]:
			input_sequence = input_sequence/self.data_std
		elif self.tt == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence)/(self.data_max-self.data_min))
		elif self.tt == "meanOnly":
			input_sequence = np.array(input_sequence)
		elif self.tt == "stdOnly":
			input_sequence = np.array(input_sequence/self.data_std)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleXdot(self, input_sequence):
		if self.tt in ["Standard", "standard", "Standard2", "standard2", "Standard3", "standard3"]:
			input_sequence = input_sequence*self.data_std.T
		elif self.tt == "MinMaxZeroOne":
			# NOTE this is incorrectly implemented, oops
			input_sequence = np.array(input_sequence*(self.data_max - self.data_min))
		elif self.tt == "meanOnly":
			input_sequence = np.array(input_sequence)
		elif self.tt == "stdOnly":
			input_sequence = np.array(input_sequence*self.data_std.T)
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def scaleM(self, input_sequence, reuse=False):
		# data_mean = np.mean(train_input_sequence,0)
		# data_std = np.std(train_input_sequence,0)
		# train_input_sequence = (train_input_sequence-data_mean)/data_std

		if self.component_wise:
			axis = None
		else:
			axis = 0

		if not reuse:
			self.derivative_mean = np.mean(input_sequence,axis=axis)
			self.derivative_std = np.std(input_sequence,axis=axis)
			self.derivative_min = np.min(input_sequence,axis=axis)
			self.derivative_max = np.max(input_sequence,axis=axis)
		if self.tt_derivative == "Standard" or self.tt_derivative == "standard":
			input_sequence = (input_sequence-self.derivative_mean)/self.derivative_std
		elif self.tt_derivative == "MinMaxZeroOne":
			input_sequence = np.array((input_sequence-self.derivative_min)/(self.derivative_max-self.derivative_min))
		elif self.tt_derivative == "meanOnly":
			input_sequence = np.array(input_sequence-self.derivative_mean)
		elif self.tt_derivative == "stdOnly":
			input_sequence = np.array(input_sequence/self.derivative_std)
		elif self.tt_derivative == "timesStd":
			input_sequence = np.array(input_sequence*self.derivative_std)
		elif self.tt_derivative != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence

	def descaleM(self, input_sequence):
		if self.tt_derivative == "Standard" or self.tt_derivative == "standard":
			input_sequence = input_sequence*self.derivative_std.T + self.derivative_mean
		elif self.tt_derivative == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.derivative_max - self.derivative_min) + self.derivative_min)
		elif self.tt_derivative == "meanOnly":
			input_sequence = np.array(input_sequence+self.derivative_mean)
		elif self.tt_derivative == "stdOnly":
			input_sequence = np.array(input_sequence*self.derivative_std.T)
		elif self.tt_derivative == "timesStd":
			input_sequence = np.array(input_sequence/self.derivative_std.T)
		elif self.tt_derivative != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence


	def descaleDataParallel(self, input_sequence, interaction_length):
		# Descaling in the parallel model requires to substract the neighboring points from the scaler
		if self.tt == "MinMaxZeroOne":
			input_sequence = np.array(input_sequence*(self.data_max[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)] - self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)]) + self.data_min[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
		elif self.tt == "Standard" or self.tt == "standard":
			input_sequence = np.array(input_sequence*self.data_std[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)].T + self.data_mean[getFirstActiveIndex(interaction_length):getLastActiveIndex(interaction_length)])
		elif self.tt != "no":
			raise ValueError("Scaler not implemented.")
		return input_sequence
