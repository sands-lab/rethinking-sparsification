import numpy as np
import random
import time
import pickle
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from scipy.stats import norm as norm_d
from scipy.stats import randint
from scipy.stats import bernoulli
from functions import *
import scipy
from copy import deepcopy

def ec_l_svrg_diana(filename, x_init, A, y, gamma, p, sparsificator, sparsificator_params, quant, quant_params, alpha,
                    data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    num_of_local_data = A[data_split[0]].shape[0]
    assert(m == num_of_workers*num_of_local_data)
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h = np.zeros(n)
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
        
    
    #this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=data_sizes[0], size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers-1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
    indices_counter = 0
    
    #it is needed for l-svrg updates
    bernoulli_arr = bernoulli.rvs(p, size=num_of_workers*1000)
    bernoulli_size = len(bernoulli_arr)
    w_vectors = np.tile(deepcopy(x), [num_of_workers,1])
    grads_w = logreg_grad(x, [A[data_split[0]], y[data_split[0]], l2, sparse_full])
    for i in range(num_of_workers-1):
        grads_w = np.vstack((grads_w, logreg_grad(x, [A[data_split[i+1]], y[data_split[i+1]], l2, sparse_full])))
    
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    bernoulli_counter = 0
    
    for it in range(int(S*num_of_local_data)):
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=data_sizes[0], size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers-1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
            indices_counter = 0
            
        if bernoulli_counter == bernoulli_size:
            bernoulli_arr = bernoulli.rvs(p, size=bernoulli_size)
            bernoulli_counter = 0
        
        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            hat_g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch]) - logreg_grad(w_vectors[i], [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch]) + grads_w[i]
            h_i = h_vectors[i]
            g_i = hat_g_i - h_i + h
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            quant_diff, q_bits_i = quant(hat_g_i - h_i, quant_params)
            h_vectors[i] = h_i + alpha * quant_diff
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
            bits_sum_temp += q_bits_i
            if (bernoulli_arr[bernoulli_counter] == 1):
                w_vectors[i] = deepcopy(x)
                grads_w[i] = logreg_grad(w_vectors[i], [A_i, y_i, l2, sparse_stoch])
                num_of_data_passes += 1.0/num_of_workers
            bernoulli_counter += 1
                
        
        v = v / num_of_workers
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        h = np.mean(h_vectors, axis=0)
        x = x - v
        
        indices_counter += 1
        num_of_data_passes += 2.0/num_of_local_data
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    
    with open("dump/"+filename+"_EC_L_SVRG_DIANA_gamma_"+str(gamma)+"_l2_"+str(l2)+"_alpha_"+str(alpha)
              +"_p_"+str(p)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+"_quantization_"+quant_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_l_svrg(filename, x_init, A, y, gamma, p, sparsificator, sparsificator_params, data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    num_of_local_data = A[data_split[0]].shape[0]
    assert(m == num_of_workers*num_of_local_data)
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
        
    
    #this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=data_sizes[0], size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers-1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
    indices_counter = 0
    
    #it is needed for l-svrg updates
    bernoulli_arr = bernoulli.rvs(p, size=num_of_workers*1000)
    bernoulli_size = len(bernoulli_arr)
    w_vectors = np.tile(deepcopy(x), [num_of_workers,1])
    grads_w = logreg_grad(x, [A[data_split[0]], y[data_split[0]], l2, sparse_full])
    for i in range(num_of_workers-1):
        grads_w = np.vstack((grads_w, logreg_grad(x, [A[data_split[i+1]], y[data_split[i+1]], l2, sparse_full])))
    
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    bernoulli_counter = 0
    
    for it in range(int(S*num_of_local_data)):
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=data_sizes[0], size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers-1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
            indices_counter = 0
            
        if bernoulli_counter == bernoulli_size:
            bernoulli_arr = bernoulli.rvs(p, size=bernoulli_size)
            bernoulli_counter = 0
        
        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch]) - logreg_grad(w_vectors[i], [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch]) + grads_w[i]
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
            if (bernoulli_arr[bernoulli_counter] == 1):
                w_vectors[i] = deepcopy(x)
                grads_w[i] = logreg_grad(w_vectors[i], [A_i, y_i, l2, sparse_stoch])
                num_of_data_passes += 1.0/num_of_workers
            bernoulli_counter += 1
                
        
        v = v / num_of_workers 
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        x = x - v
        
        indices_counter += 1
        num_of_data_passes += 2.0/num_of_local_data
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    
    with open("dump/"+filename+"_EC_L_SVRG_gamma_"+str(gamma)+"_l2_"+str(l2)+"_p_"+str(p)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_diana_sgd(filename, x_init, A, y, gamma, sparsificator, sparsificator_params, quant, quant_params, alpha,
                data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    num_of_local_data = A[data_split[0]].shape[0]
    assert(m == num_of_workers*(A[data_split[0]].shape[0]))
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h = np.zeros(n)
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
        
        
    #this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=data_sizes[0], size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers-1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
    indices_counter = 0
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    
    for it in range(S*num_of_local_data):      
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=data_sizes[0], size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers-1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
            indices_counter = 0
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            hat_g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch])
            h_i = h_vectors[i]
            g_i = hat_g_i - h_i + h
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            quant_diff, q_bits_i = quant(hat_g_i - h_i, quant_params)
            h_vectors[i] = h_i + alpha * quant_diff
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
            bits_sum_temp += q_bits_i
        
        v = v / num_of_workers
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        h = np.mean(h_vectors, axis=0)
        x = x - v
        
        indices_counter += 1
        num_of_data_passes += 1.0/num_of_local_data
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    
    with open("dump/"+filename+"_EC_DIANA_SGD_gamma_"+str(gamma)+"_alpha_"+str(alpha)
              +"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+"_quantization_"+quant_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_diana_gd(filename, x_init, A, y, gamma, sparsificator, sparsificator_params, quant, quant_params, alpha,
                data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    assert(m == num_of_workers*(A[data_split[0]].shape[0]))
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    h = np.zeros(n)
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    
    for it in range(S):        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            hat_g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
            h_i = h_vectors[i]
            g_i = hat_g_i - h_i + h
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            quant_diff, q_bits_i = quant(hat_g_i - h_i, quant_params)
            h_vectors[i] = h_i + alpha * quant_diff
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
            bits_sum_temp += q_bits_i
        
        v = v / num_of_workers
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        h = np.mean(h_vectors, axis=0)
        x = x - v
        
        num_of_data_passes += 1.0
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    
    with open("dump/"+filename+"_EC_DIANA_GD_gamma_"+str(gamma)+"_alpha_"+str(alpha)
              +"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+"_quantization_"+quant_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_gd_star_const_stepsize(filename, x_init, A, y, gamma, sparsificator, sparsificator_params, data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    assert(m == num_of_workers*(A[data_split[0]].shape[0]))
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    
    for it in range(S):        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch]) - logreg_grad(x_star, [A_i, y_i, l2, sparse_stoch])
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
        
        v = v / num_of_workers 
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        x = x - v
        
        num_of_data_passes += 1.0
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    with open("dump/"+filename+"_EC_GD_star_const_stepsize_gamma_"+str(gamma)+"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_gd_const_stepsize(filename, x_init, A, y, gamma, sparsificator, sparsificator_params, data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    assert(m == num_of_workers*(A[data_split[0]].shape[0]))
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    
    for it in range(S):        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            g_i = logreg_grad(x, [A_i, y_i, l2, sparse_stoch])
            e_i = error_vectors[i]
            v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            v += v_i
            avg_error_norm += norm(e_i)**2
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            bits_sum_temp += bits_i
        
        v = v / num_of_workers 
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        x = x - v
        
        num_of_data_passes += 1.0
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms}
    
    with open("dump/"+filename+"_EC_GD_const_stepsize_gamma_"+str(gamma)+"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res

def ec_sgd_const_stepsize(filename, x_init, A, y, gamma, sparsificator, sparsificator_params, data_split,  
         l2=0, sparse_full=True, sparse_stoch=False, S=50, max_t=np.inf,
         batch_size=1, save_info_period=100, x_star=None, f_star=None):
    #m -- total number of datasamples
    #n -- dimension of the problem
    m, n = A.shape
    assert(len(x_init) == n)
    assert(len(y) == m)
    
    if x_star is None:
        x_star = np.zeros(n)
    if f_star is None:
        f_star = 0
    ref_point = np.array(x_star)
    x = np.array(x_init)
    
    num_of_workers = len(data_split)
    num_of_local_data = A[data_split[0]].shape[0]
    assert(m == num_of_workers*num_of_local_data)
    error_vectors = np.tile(np.zeros(n), [num_of_workers,1])
    data_sizes = np.array([])
    for i in range(num_of_workers):
        data_sizes = np.append(data_sizes, len(data_split[i]))
    
    #this array below is needed to reduce the time of sampling stochastic gradients
    indices_arr = randint.rvs(low=0, high=data_sizes[0], size=1000)
    num_of_indices = len(indices_arr)
    for i in range(num_of_workers-1):
        indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
    indices_counter = 0
    
    its = np.array([0])
    tim = np.array([0.0])
    data_passes = np.array([0.0])
    func_val = np.array([F(x, [A, y, l2, sparse_full, 0])-f_star])
    sq_distances = np.array([norm(x - ref_point) ** 2])
    number_of_bits = np.array([0]) #counts the number of bits per worker
    avg_ecgrad_norms = np.array([0]) # average error-compensated gradient norm (across workers)
    avg_grad_norms = np.array([0]) # average gradient norm (across workers)
    avg_error_norms = np.array([0]) # average error norm (across workers)
    
    t_start = time.time()
    num_of_data_passes = 0.0
    num_of_bits = 0.0
    
    if sparse_stoch:
        A_for_batch = A
    else:
        A_for_batch = A.toarray()
    
    indices_counter = 0
    
    # For DCT
    previous_thr = [np.nan]*num_of_workers
    # For DGC, initial sparsity is 75%
    previous_k = [int(0.25*n)]*num_of_workers
    
    for it in range(int(S*num_of_local_data)):
        #print(it)
        if indices_counter == num_of_indices:
            indices_arr = randint.rvs(low=0, high=data_sizes[0], size=num_of_indices)
            num_of_indices = len(indices_arr)
            for i in range(num_of_workers-1):
                indices_arr = np.vstack((indices_arr, randint.rvs(low=0, high=data_sizes[i+1], size=num_of_indices)))
            indices_counter = 0
        
        #below we emulate the workers behavior and aggregate their updates on-the-fly
        v = np.zeros(n)
        avg_ecgrad_norm = 0
        avg_grad_norm = 0
        bits_sum_temp = 0
        avg_error_norm = 0
        for i in range(num_of_workers):
            A_i = A_for_batch[data_split[i]]
            y_i = y[data_split[i]]
            g_i = logreg_grad(x, [A_i[indices_arr[i][indices_counter:indices_counter+1]], 
                                  y_i[indices_arr[i][indices_counter:indices_counter+1]], l2, sparse_stoch])
            e_i = error_vectors[i]
            avg_error_norm += norm(e_i)**2
            if sparsificator == dct:
                # DCT requires addtional arguments as current iteration count and previous threshold value
                v_i, bits_i, previous_thr[i] = sparsificator(e_i+gamma*g_i, sparsificator_params+[it, previous_thr[i]])
            elif sparsificator == dgc:
                # DGC requires additional arguments as current iteration count and previous k
                v_i, bits_i, previous_k[i] = sparsificator(e_i+gamma*g_i, sparsificator_params+[it, previous_k[i]])       
            else:
                v_i, bits_i = sparsificator(e_i+gamma*g_i, sparsificator_params)
            error_vectors[i] = e_i + gamma*g_i - v_i
            v += v_i
            avg_ecgrad_norm += norm(e_i + gamma*g_i)**2
            avg_grad_norm += norm(gamma*g_i)**2
            avg_ecgrad_topk += np.sort(np.abs(e_i + gamma*g_i))[-10]
            bits_sum_temp += bits_i
        
        v = v / num_of_workers 
        avg_ecgrad_norm = avg_ecgrad_norm / num_of_workers
        avg_error_norm = avg_error_norm / num_of_workers
        avg_grad_norm = avg_grad_norm / num_of_workers
        avg_ecgrad_topk = avg_ecgrad_topk / num_of_workers
        x = x - v
        
        indices_counter += 1
        num_of_data_passes += 1.0/num_of_local_data
        num_of_bits += bits_sum_temp*1.0/num_of_workers #we count number of bits per worker
        if ((it + 1) % save_info_period == 0):
            its = np.append(its, it + 1)
            tim = np.append(tim, time.time() - t_start)
            data_passes = np.append(data_passes, num_of_data_passes)
            func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
            sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
            number_of_bits = np.append(number_of_bits, num_of_bits)
            avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
            avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
            avg_error_norms = np.append(avg_error_norms, avg_error_norm)
            avg_ecgrad_topks = np.append(avg_ecgrad_topks, avg_ecgrad_topk)
            
        if tim[-1] > max_t:
            break
    
    if ((it + 1) % save_info_period != 0):
        its = np.append(its, it + 1)
        tim = np.append(tim, time.time() - t_start)
        data_passes = np.append(data_passes, num_of_data_passes)
        func_val = np.append(func_val, F(x, [A, y, l2, sparse_full, 0])-f_star)
        sq_distances = np.append(sq_distances, norm(x - ref_point) ** 2)
        number_of_bits = np.append(number_of_bits, num_of_bits)
        avg_ecgrad_norms = np.append(avg_ecgrad_norms, avg_ecgrad_norm)
        avg_grad_norms = np.append(avg_grad_norms, avg_grad_norm)
        avg_error_norms = np.append(avg_error_norms, avg_error_norm)
        avg_ecgard_topk = np.append(avg_ecgrad_topks, avg_ecgrad_topk)
        
        
    res = {'last_iter':x, 'func_vals':func_val, 'iters':its, 'time':tim, 'data_passes':data_passes,
           'squared_distances':sq_distances, 'bits':number_of_bits, 'avg_ecgrad_norms':avg_ecgrad_norms, 'avg_grad_norms':avg_grad_norms, 'avg_error_norms':avg_error_norms, 'avg_ecgrad_topks':avg_ecgrad_topks}
    
    with open("dump/"+filename+"_EC_SGD_const_stepsize_gamma_"+str(gamma)+"_l2_"+str(l2)+"_num_of_epochs_"+str(S)
              +"_num_of_workers_"+str(num_of_workers)+"_sparsificator_"
              +sparsificator_params[0]+".txt", 'wb') as file:
        pickle.dump(res, file)
    return res





