import numpy as np
from numpy.linalg import norm
import pickle
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm as norm_d
from scipy.stats import expon
from scipy.stats import weibull_min as weibull
from scipy.stats import burr12 as burr
from scipy.stats import randint
from scipy.stats import uniform
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.signal import savgol_filter
import copy
import math
import time
from scipy.optimize import minimize
from scipy.sparse.linalg import svds
from scipy.linalg import svdvals
import scipy
from sklearn.datasets import load_svmlight_file
import pickle
from pathlib import Path



def prepare_data(dataset):
    filename = "datasets/" + dataset + ".txt"

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    
    if (2 in y) & (1 in y):
        y = 2 * y - 3
    if (2 in y) & (4 in y):
        y = y - 3
    assert((-1 in y) & (1 in y))
    
    sparsity_A = A.count_nonzero() / (m * n)
    return A, y, m, n, sparsity_A


def prepare_data_distrib(dataset, data_size, num_of_workers):
    filename = "datasets/" + dataset + ".txt"

    data = load_svmlight_file(filename)
    A, y = data[0], data[1]
    m, n = A.shape
    assert(data_size <= m)
    
    size_of_local_data = int(data_size*1.0 / num_of_workers)
    A = A[0:size_of_local_data*num_of_workers]
    y = y[0:size_of_local_data*num_of_workers]
    m, n = A.shape
    assert(data_size == size_of_local_data*num_of_workers)
    
    perm = np.random.permutation(m)
    data_split = perm[0:size_of_local_data]
    for i in range(num_of_workers-1):
        data_split = np.vstack((data_split, perm[(i+1)*size_of_local_data:(i+2)*size_of_local_data]))
    
    if (2 in y) & (1 in y):
        y = 2 * y - 3
    if (2 in y) & (4 in y):
        y = y - 3
    assert((-1 in y) & (1 in y))
    
    sparsity_A = A.count_nonzero() / (m * n)
    return A, y, m, n, sparsity_A, data_split


def compute_L(dataset, A):
    filename = "dump/"+dataset+"_L.txt"
    file_path = Path(filename)
    if file_path.is_file():
        with open(filename, 'rb') as file:
            L, average_L, worst_L = pickle.load(file)
    else:
        sigmas = svds(A, return_singular_vectors=False)
        m = A.shape[0]
        L = sigmas.max()**2 / (4*m)
        
        worst_L = 0
        average_L = 0
        denseA = A.toarray()
        for i in range(m):
            L_temp = (norm(denseA[i])**2)*1.0 / 4
            average_L += L_temp / m
            if L_temp > worst_L:
                worst_L = L_temp
        with open(filename, 'wb') as file:
            pickle.dump([L, average_L, worst_L],file)
    return L, average_L, worst_L

def compute_L_distrib(dataset, A):
    filename = "dump/"+dataset+"_"+str(A.shape[0])+"_"+"_L.txt"
    file_path = Path(filename)
    if file_path.is_file():
        with open(filename, 'rb') as file:
            L, average_L, worst_L = pickle.load(file)
    else:
        sigmas = svds(A, return_singular_vectors=False)
        m = A.shape[0]
        L = sigmas.max()**2 / (4*m)
        
        worst_L = 0
        average_L = 0
        denseA = A.toarray()
        for i in range(m):
            L_temp = (norm(denseA[i])**2)*1.0 / 4
            average_L += L_temp / m
            if L_temp > worst_L:
                worst_L = L_temp
        with open(filename, 'wb') as file:
            pickle.dump([L, average_L, worst_L],file)
    return L, average_L, worst_L

def save_split(dataset, num_of_workers, data_split):
    filename = "dump/"+dataset+"_split_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump(data_split, file)
        
def read_split(dataset, num_of_workers):
    filename = "dump/"+dataset+"_split_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_problem(dataset, num_of_workers, params):
    filename = "dump/"+dataset+"_problem_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump(params, file)
        
def read_problem(dataset, num_of_workers):
    filename = "dump/"+dataset+"_problem_num_of_workers_"+str(num_of_workers)+".txt"
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save_solution(dataset, l2, l1, x_star, f_star):
    filename = "dump/"+dataset+"_solution_l2_"+str(l2)+"_l1_"+str(l1)+".txt"
    with open(filename, 'wb') as file:
        pickle.dump([x_star, f_star], file)

def read_solution(dataset, l2, l1):
    with open('dump/'+dataset+'_solution_l2_'+str(l2)+"_l1_"+str(l1)+".txt", 'rb') as file:
        return pickle.load(file)


def read_results_from_file(filename, method, args):
    if method == "EC_SGD_const_stepsize":
        with open('dump/'+filename+'_EC_SGD_const_stepsize_gamma_'+str(args[0])+"_l2_"+str(args[1])
                  +"_num_of_epochs_"+str(args[2])+"_num_of_workers_"+str(args[3])
                  +"_sparsificator_"+args[4]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_L-SVRG-DIANA":
        with open('dump/'+filename+'_EC_L_SVRG_DIANA_gamma_'+str(args[0])+"_l2_"+str(args[1]) +"_alpha_"+str(args[2])
                  +"_p_"+str(args[3])
                  +"_num_of_epochs_"+str(args[4])+"_num_of_workers_"+str(args[5])
                  +"_sparsificator_"+args[6]+"_quantization_"+args[7]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_L-SVRG":
        with open('dump/'+filename+'_EC_L_SVRG_gamma_'+str(args[0])+"_l2_"+str(args[1]) +"_p_"+str(args[2])
                  +"_num_of_epochs_"+str(args[3])+"_num_of_workers_"+str(args[4])
                  +"_sparsificator_"+args[5]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_GD_const_stepsize":
        with open('dump/'+filename+'_EC_GD_const_stepsize_gamma_'+str(args[0])+"_l2_"+str(args[1])
                  +"_num_of_epochs_"+str(args[2])+"_num_of_workers_"+str(args[3])
                  +"_sparsificator_"+args[4]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_GD_star_const_stepsize":
        with open('dump/'+filename+'_EC_GD_star_const_stepsize_gamma_'+str(args[0])+"_l2_"+str(args[1])
                  +"_num_of_epochs_"+str(args[2])+"_num_of_workers_"+str(args[3])
                  +"_sparsificator_"+args[4]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_DIANA_GD":
        with open('dump/'+filename+'_EC_DIANA_GD_gamma_'+str(args[0])+"_alpha_"+str(args[1])
                  +"_l2_"+str(args[2])
                  +"_num_of_epochs_"+str(args[3])+"_num_of_workers_"+str(args[4])
                  +"_sparsificator_"+args[5]+"_quantization_"+args[6]+".txt", 'rb') as file:
            return pickle.load(file)
    if method == "EC_DIANA_SGD":
        with open('dump/'+filename+'_EC_DIANA_SGD_gamma_'+str(args[0])+"_alpha_"+str(args[1])
                  +"_l2_"+str(args[2])
                  +"_num_of_epochs_"+str(args[3])+"_num_of_workers_"+str(args[4])
                  +"_sparsificator_"+args[5]+"_quantization_"+args[6]+".txt", 'rb') as file:
            return pickle.load(file)

def make_plots(args):
    supported_modes_y = ['func_vals', 'squared_distances', 'bits', 'avg_ecgrad_norms', 'avg_grad_norms', 'avg_error_norms', 'avg_ecgrad_topks', 'total_bits', 'non-zero-density']
    supported_modes_x = ['time', 'data_passes', 'iters', 'bits']
    
    filename = args[0]
    mode_y = args[1]
    mode_x = args[2]
    figsize = args[3]
    sizes = args[4]
    title = args[5]
    methods = args[6]
    bbox_to_anchor = args[7]
    legend_loc = args[8]
    save_fig = args[9]
    
    
    title_size = sizes[0]
    linewidth = sizes[1]
    markersize = sizes[2]
    legend_size = sizes[3]
    xlabel_size = sizes[4]
    ylabel_size = sizes[5]
    xticks_size = sizes[6]
    yticks_size = sizes[7]
    
    assert(mode_y in supported_modes_y)
    assert(mode_x in supported_modes_x)
    
    fig = plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_size)
    marker = itertools.cycle(('+', 'd', 'x', 'o', '^', 's', '*', 'p', '<', '>', '^'))
    color = itertools.cycle((('tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown')))
    num_of_methods = len(methods)
    for idx, method in enumerate(methods):
        res = read_results_from_file(filename, method[0], method[1])
        if method[3] == None:
            length = len(res['iters'])
        else:
            length = method[3]
        #print("Length=", length)
        if mode_y == 'avg_ecgrad_norms' or mode_y == 'avg_grad_norms' or mode_y == 'avg_error_norms' or mode_y == 'avg_ecgrad_topks': 
            plt.semilogy(res[mode_x][1:length], res[mode_y][1:length], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int((length-1)/(10*num_of_methods)), len(res[mode_x][1:length]), int((length-1)/10)), 
                label = method[2], color=next(color))
#             plt.plot(res[mode_x][1:length], res[mode_y][1:length], linewidth=linewidth, marker=next(marker), 
#                 markersize = markersize, 
#                 markevery=range(-idx*int((length-1)/(10*num_of_methods)), len(res[mode_x][1:length]), int((length-1)/10)), 
#                 label = method[2], color=next(color))
        elif mode_y == 'bits':
            bits_prev = np.insert(res[mode_y][0:length-1], 0, 0, axis=0)
            plt.plot(res[mode_x][1:length], res[mode_y][1:length] - bits_prev[1:length], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int((length-1)/(10*num_of_methods)), len(res[mode_x][1:length]), int((length-1)/10)), 
                label = method[2], color=next(color))
        elif mode_y == 'non-zero-density':
            min_length=0
            bits_prev = np.insert(res['bits'][0:length-1], 0, 0, axis=0)
            plt.scatter(res[mode_x][min_length:length], (res['bits'][min_length:length] - bits_prev[min_length:length]!=0).astype(int), linewidth=linewidth,s=markersize, label = method[2], color=next(color))
        elif mode_y == 'squared_distances':
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.semilogy(res[mode_x][0:length], res[mode_y][0:length], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                label = method[2], color=next(color))
        elif mode_x == 'bits':
            print("Initial "+mode_y+" is:", res[mode_y][0])
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.semilogy(res[mode_x][0:length], res[mode_y][0:length] / res[mode_y][0], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                label = method[2], color=next(color))
        else:
            print("Initial "+mode_y+" is:", res[mode_y][0])
            plt.semilogy(res[mode_x][0:length], res[mode_y][0:length], linewidth=linewidth, marker=next(marker), 
                markersize = markersize, 
                markevery=range(-idx*int(length/(10*num_of_methods)), len(res[mode_x][0:length]), int(length/10)), 
                label = method[2], color=next(color))
        
    
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=legend_loc, fontsize=legend_size)
    if mode_x == 'bits':
        plt.xlabel(r"Number of bits per worker", fontsize=xlabel_size)
    if mode_x == 'time':
        plt.xlabel(r"Time, $s$", fontsize=xlabel_size)
    if mode_x == 'data_passes':
        plt.xlabel(r"Epoch Number", fontsize=xlabel_size)
    if mode_x == 'iters':
        plt.xlabel(r"Iteration Number", fontsize=xlabel_size)
    if mode_y == 'func_vals':
        #plt.ylabel(r"$\frac{f(x_t)-f(x_*)}{f(x_0)-f(x_*)}$", fontsize=ylabel_size)
        plt.ylabel(r"$f(x_t)-f(x_*)$", fontsize=ylabel_size)
    if mode_y == 'squared_distances':
        plt.ylabel(r"$||x_t - x_*||_2^2$", fontsize=ylabel_size)
    if mode_y == 'bits':
        plt.ylabel(r"$k_t$", fontsize=xlabel_size)
    if mode_y == 'avg_ecgrad_norms':
        plt.ylabel(r"$\frac{1}{n}\sum_{i=1}^n||p^i_t||^2$", fontsize=xlabel_size)
    if mode_y == 'avg_grad_norms':
        plt.ylabel(r"$\frac{1}{n}\sum_{i=1}^n||\gamma_t g^i_t||^2$", fontsize=xlabel_size)
    if mode_y == 'avg_error_norms':
        plt.ylabel(r"$\frac{1}{n}\sum_{i=1}^n||e^i_t||^2$", fontsize=xlabel_size)
    if mode_y == 'avg_ecgrad_topks':
        plt.ylabel(r"$\frac{1}{n}\sum_{i=1}^n$ Top-10$(|e^i_t+\gamma_t g^i_t|)$", fontsize=xlabel_size)
    if mode_y == 'non-zero-density':
        plt.ylabel(r"$k_t \neq 0$", fontsize=xlabel_size)
    
    plt.grid(True, linewidth=0.5, linestyle='-')
    plt.xticks(fontsize=xticks_size)
    _ = plt.yticks(fontsize=yticks_size)
    
    ax = fig.gca()
    ax.xaxis.offsetText.set_fontsize(xlabel_size - 2)
    ax.yaxis.offsetText.set_fontsize(ylabel_size - 2)
    
    if save_fig[0]:
        plt.savefig("plot/"+save_fig[1], bbox_inches='tight')
        

