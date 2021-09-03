#!/usr/bin/env python
import os
import numpy as np
from scipy.linalg import svdvals, eigvals
from scipy.sparse.linalg import svds as sparse_svds
from scipy.sparse.linalg import eigs as sparse_eigs
import itertools

# Plotting parameters
import matplotlib
import pandas as pd
matplotlib.use("Agg")

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
import pickle
from matplotlib import colors
from matplotlib.colors import Normalize
import six
from scipy.interpolate import interpn
color_dict = dict(six.iteritems(colors.cnames))

# font = {'size': 16}
# matplotlib.rc('font', **font)

# sns.set(rc={'text.usetex': True}, font_scale=4)


import pdb

def make_time_plot(df, x_name='t_warmup', y_name='mse_total'):
	colors = {'EnKF': 'orange', 'ad hoc': 'blue',
			  'True': 'black', '3DVAR': 'green',
			  'knn': 'purple',
			  'torch.opt': 'magenta'}




def plot_trajectories(times, traj_dict, fig_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    line_vec = {'EnKF': ':', 'Ad hoc': '--', '3DVAR': '-.', '3DVAR pred': '-.'}

    for key in traj_dict:
        if key=='True':
            ax.plot(times, traj_dict[key], linestyle='-', color='black', linewidth=4, label=key)
        else:
            ax.plot(times, traj_dict[key], linestyle=line_vec[key], linewidth=4, label=key)
    ax.set_title('State Synchronization')
    ax.set_ylabel('r')
    ax.set_xlabel('time')
    ax.legend()
    plt.savefig(fig_path)
    plt.close()

def plot_assimilation_residual_statistics(res, fig_path):
    # plot sequence
    n_vars = res.shape[1]
    fig, ax = plt.subplots(nrows=n_vars, ncols=n_vars, figsize=(12,6))
    for i in range(n_vars): # row
        for j in range(n_vars): # col
            axfoo = ax[i,j]
            if i > j:
                density_scatter(res[:,i], res[:,j], ax = axfoo, s=5)
            elif i==j:
                sns.kdeplot(res[:,i], ax=axfoo, linewidth=4)
            else:
                axfoo.axis('off')
                continue

            if i==(n_vars-1):
                axfoo.set_xlabel('X_{}'.format(j))
            if j==0:
                axfoo.set_ylabel('X_{}'.format(i))

    fig.suptitle('Bivariate statistics')
    plt.savefig(fig_path)
    for i in range(n_vars): # row
        for j in range(n_vars): # col
            ax[i,j].set_yscale('symlog')
    plt.savefig(fig_path + '_ylog')

    plt.close()

def plot_loss(times, loss, fig_path):
    # plot sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.plot(times, loss, linestyle='-', linewidth=4)
    ax.set_title('Loss sequence')
    ax.set_ylabel('Loss')
    ax.set_xlabel('time')
    plt.savefig(fig_path)
    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog')

def plot_K_learning(times, K_vec, fig_path):
    # plot sequence
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    for i in range(K_vec.shape[1]):
        for j in range(K_vec.shape[2]):
            ax.plot(times, K_vec[:,i,j], linestyle='-', linewidth=4, label='K_{i}{j}'.format(i=i,j=j))
    ax.set_title('K learning')
    ax.set_ylabel('K')
    ax.set_xlabel('time')
    ax.legend()
    plt.savefig(fig_path)

def plot_assimilation_errors(times, error_dict, eps, fig_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    eps_vec = [eps for _ in range(len(times))]
    line_vec = {'EnKF': ':', 'Ad hoc': '--', '3DVAR': '-.', '3DVAR pred': '-.'}

    for key in error_dict:
        ax.plot(times, error_dict[key], linestyle=line_vec[key], linewidth=4, label=key)
    ax.plot(times, eps_vec, linestyle='--', linewidth=4, color='black', label = r'$\epsilon$')
    ax.set_title('State assimilation error convergence')
    ax.set_ylabel('MSE')
    ax.set_xlabel('time')
    ax.legend()
    # plt.savefig(fig_path)

    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog')

    ax.set_xscale('log')
    plt.savefig(fig_path + '_ylog_xlog')

    plt.close()



def density_scatter( x , y, ax = None, sort = True, bins = 20, do_cbar=False, n_subsample=None, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    """
    if ax is None :
        fig , ax = plt.subplots()

    if n_subsample:
        inds = np.random.choice(len(x), replace=False, size=n_subsample)
        x = x[inds]
        y = y[inds]
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    if do_cbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')

    return ax



def box(df, output_dir, metric_list, x="model_name", fname_shape='summary_eps_{}', figsize=(24, 20)):
    for metric in metric_list:
        try:
            fig_path = os.path.join(output_dir, fname_shape.format(metric))
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
            sns.boxplot(ax=ax, data=df, x=x, y=metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='large')

            plt.savefig(fig_path)
            ax.set_yscale('log')
            plt.savefig(fig_path + '_ylog')

        except:
            print('Failed at', metric)
            pass
        plt.close()

def new_box(df, fig_path,
            x='Model',
            y='t_valid_005',
            order=None,
            figsize=(12,10),
            fontsize=20,
            ylabel=None,
            xlabel=None,
            title=None,
            rotation=20,
            legloc='upper right',
            ax=None):

    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    if ax is None:
        return_ax = False
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    else:
        return_ax = True

    sns.boxplot(ax=ax, data=df, x=x, y=y, order=order)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, horizontalalignment='right', fontsize='large')
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=fontsize)
    # ax.legend(loc=legloc, fontsize=fontsize)

    # fig.subplots_adjust(wspace=0.3, hspace=0.3)

    if return_ax:
        ax.set_yscale('log')
        return ax
    else:
        fig.subplots_adjust(bottom=0.15, left=0.15)
        plt.savefig(fig_path)
        ax.set_yscale('log')
        plt.savefig(fig_path + '_ylog')
        plt.close()


def new_summary(df, fig_path, hue='Model', style='Uses $f_0$', x="$\epsilon$", y='t_valid_005',
                figsize=(12,10),
                fontsize=20,
                ylabel=None,
                xlabel=None,
                title=None,
                estimator=np.median,
                ci='sd',
                legloc='upper right'):

    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    sns.lineplot(ax=ax, data=df, estimator=estimator, markers=True, markersize=14, ci=ci, x=x, y=y, style=style, hue=hue, err_style='band', linewidth=4)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(loc=legloc, fontsize=fontsize)
    plt.savefig(fig_path)

    ax.set_yscale('log')
    plt.savefig(fig_path + '_ylog')

    ax.set_xscale('log')
    plt.savefig(fig_path + '_xlog_ylog')

    ax.set_yscale('linear')
    plt.savefig(fig_path + '_xlog')

    plt.close()


def summarize(df, hue, style, output_dir, metric_list, x="f0eps", fname_shape='summary_eps_{}', figsize=(24, 12)):
    for metric in metric_list:
        try:
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize)
            sns.lineplot(ax=ax, data=df, x=x, y=metric, style=style, hue=hue, err_style='bars', linewidth=4, ci='sd')
            fig_path = os.path.join(output_dir, fname_shape.format(metric))
            plt.savefig(fig_path)
            ax.set_yscale('log')
            plt.savefig(fig_path + '_ylog')

            ax.set_xscale('log')
            plt.savefig(fig_path + '_xlog_ylog')

            ax.set_yscale('linear')
            plt.savefig(fig_path + '_xlog')
        except:
            print('Failed at', metric)
            pass
        plt.close()

def plotMatrixSpectrum(model, A, mat_name):
    fig_path = os.path.join(model.fig_dir, "singular_values_{:}.png".format(mat_name))
    try:
        s = svdvals(A)
    except:
        s = -np.sort(-sparse_svds(A, return_singular_vectors=False, k=min(100,min(A.shape))))
    plt.plot(s,'o')
    plt.ylabel(r'$\sigma$')
    plt.title('Singular values of {:}'.format(mat_name))
    plt.savefig(fig_path)
    plt.close()

    if A.shape[0]==A.shape[1]: #is square
        fig_path = os.path.join(model.fig_dir, "eigenvalues_{:}.png".format(mat_name))
        try:
            eig = eigvals(A)
        except:
            eig = sparse_eigs(A, return_eigenvectors=False, k=min(1000,min(A.shape)))
        plt.plot(eig.real, eig.imag, 'o')
        plt.xlabel(r'Re($\lambda$)')
        plt.ylabel(r'Im($\lambda$)')
        plt.title('Eigenvalues of {:}'.format(mat_name))
        plt.savefig(fig_path)
        plt.close()

def plotMatrix(model, A, mat_name):
    fig_path = os.path.join(model.fig_dir, "matrix_{:}.png".format(mat_name))
    # plot matrix visualizations
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
    foo = ax[0].matshow(A, vmin=np.min(A), vmax=np.max(A), aspect='auto')
    # ax[0].axes.xaxis.set_visible(False)
    # ax[0].axes.yaxis.set_visible(False)
    ax[0].set_title(mat_name)
    fig.colorbar(foo, ax=ax[0])

    sns.ecdfplot(data=np.abs(A).reshape(-1,1), ax=ax[1])
    ax[1].set_xscale('log')
    ax[1].set_title('Distribution of matrix entries')
    plt.savefig(fig_path)
    plt.close()


def plot_model_characteristics(figdir, X, fontsize=20):
    os.makedirs(figdir, exist_ok=True)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    X = np.squeeze(X)
    # Pool data and plot Inv Measure
    fig_path = os.path.join(figdir, "inv_stats_POOL.png")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 14))
    sns.kdeplot(X.reshape(-1), ax=ax, linewidth=4)
    plt.savefig(fig_path)
    plt.close()

    # Plot Invariant Measure of each state individually as a marginal
    fig_path = os.path.join(figdir, "inv_stats_MARGINAL.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=1, ncols=ndim, figsize=(24, 14))
    for x_in in range(ndim): #
        sns.kdeplot(X[:,x_in], ax=ax[x_in], linewidth=4)
        ax[x_in].set_xlabel('X_{}'.format(x_in))
    plt.savefig(fig_path)
    plt.close()

    # Plot Invariant Measure of each state individually, plus bivariate scatter
    fig_path = os.path.join(figdir, "inv_stats_BIVARIATE.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
    # x axis is INPUT dim for model
    # y axis is OUTPUT dim for model
    for x_in in range(ndim): #
        for x_out in range(ndim):
            if x_out==x_in:
                sns.kdeplot(X[:,x_in], ax=ax[x_out][x_in], linewidth=4)
                # ax[x_out][x_in].ksdensity(X[:,x_in])
            else:
                # sns.scatter(x=X[:,x_in], y=X[:,x_out], ax=ax[x_out][x_in])
                ax[x_out][x_in].scatter(X[:,x_in], X[:,x_out])
            ax[x_out][x_in].set_xlabel('X_{}'.format(x_in))
            ax[x_out][x_in].set_ylabel('X_{}'.format(x_out))
    plt.savefig(fig_path)
    plt.close()


def plot_io_characteristics(figdir, X, y=None, gpr_predict=None, fontsize=20):

    if y is None:
        y = gpr_predict(X)

    os.makedirs(figdir, exist_ok=True)
    font = {'size': fontsize}
    matplotlib.rc('font', **font)

    # Plot bivariate scatter
    fig_path = os.path.join(figdir, "bivariate_scatter.png")
    ndim = X.shape[1]
    fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
    # x axis is INPUT dim for model
    # y axis is OUTPUT dim for model
    for x_in in range(ndim): #
        for x_out in range(ndim):
            ax[x_out][x_in].scatter(X[:,x_in], y[:,x_out])
            ax[x_out][x_in].set_xlabel('Xin_{}'.format(x_in))
            ax[x_out][x_in].set_ylabel('Yout_{}'.format(x_out))
    plt.savefig(fig_path)
    plt.close()

    # Plot fancier thing
    for y_out in range(ndim):
        fig_path = os.path.join(figdir, "contour_{}.png".format(y_out))
        fig, ax = plt.subplots(nrows=ndim, ncols=ndim, figsize=(14, 14))
        # x axis is INPUT dim for model
        # y axis is OUTPUT dim for model
        for x_in1 in range(ndim): #
            for x_in2 in range(ndim):
                # xxin1, xxin2 = np.meshgrid(X[x_in1], X[_in2])
                # ax[x_in2][x_in1].contourf(xxin1, xxin2, np.squeeze(y[:,y_out]))
                ax[x_in2][x_in1].scatter(x=X[:,x_in1], y=X[:,x_in2], c=np.squeeze(y[:,y_out]), alpha=0.5)
                ax[x_in2][x_in1].set_xlabel('Xin_{}'.format(x_in1))
                ax[x_in2][x_in1].set_ylabel('Xin_{}'.format(x_in2))
        fig.suptitle('GP field for output state {}'.format(y_out))
        plt.savefig(fig_path)
        plt.close()

    # Plot 1 big good plot for paper
    # get combinations
    fig_path = os.path.join(figdir, "contour_all.png")
    ax_combs = list(itertools.combinations(np.arange(ndim),2))

    fig, ax = plt.subplots(nrows=ndim, ncols=len(ax_combs), figsize=(14, 14))
    for y_out in range(ndim):
        # plot permutations
        cc = -1
        for x_in1, x_in2 in ax_combs:
            cc += 1
            cbar = ax[y_out][cc].scatter(x=X[:,x_in1], y=X[:,x_in2], c=np.squeeze(y[:,y_out]), alpha=0.5)
            ax[y_out][cc].set_xlabel(r'$X^{{in}}_{0}$'.format(x_in1), fontstyle='italic')
            ax[y_out][cc].set_ylabel(r'$X^{{in}}_{0}$'.format(x_in2), fontstyle='italic', rotation=0)
            ax[y_out][cc].set_title(r'$\mathbf{{X^{{out}}_{0}}}$'.format(y_out), fontweight='bold', fontsize=24)
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.colorbar(cbar, ax=ax[y_out][cc])
    plt.savefig(fig_path)
    plt.close()
