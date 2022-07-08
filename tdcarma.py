#%% Import libraries
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from pymultinest.solve import solve
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import scipy
import os
try: os.mkdir('nest_output')
except OSError: pass
import pickle
import time
#from mpi4py import MPI
plt.style.use('tableau-colorblind10')


#%% Pickle import function
def import_pickle(data):
    '''
    Import pickle data set "data" (type str)
    loaded_list contains the simulation parameters ([0]) and  dataset ([1])
    '''
    open_file = open(data, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

#%% Import data
data_dir = './data/'
data_file = 'j1206.csv'
data = pd.read_csv(data_dir+data_file)

#%% Extract data
t = data.iloc[:,0] - min(data.iloc[:,0])
y = data.iloc[:,1]
z = data.iloc[:,3]
err_A = data.iloc[:,2]**2
err_B = data.iloc[:,4]**2

#Pre-processing for flux data (TDC data)
# t = data['time'] - min(data['time'])
# y = -2.5*np.log10(data['lc_A'])
# z = -2.5*np.log10(data['lc_B'])
# err_A = (data['err_A']*2.5/data['lc_A']/np.log(10))
# err_B = (data['err_B']*2.5/data['lc_B']/np.log(10))

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 8), dpi=500)


plt.errorbar(t, y, yerr=err_A, label = 'A')
plt.errorbar(t, z, yerr=err_B, label = 'B')
plt.plot(t, y, 'k.')
plt.plot(t, z, 'k.')
plt.xlabel('Time [days]')
plt.ylabel('Magnitude [mag]')
plt.legend()
plt.show()

#%% Compute bounds on frequencies supported by data
dt = np.diff(t)
max_freq = 1 / min(dt)
min_freq = 1 / (max(t) - min(t))
C_1 = 2 * np.pi * max_freq
C_2 = 2 * np.pi * min_freq

#%% Choose order of model to fit
m = 4
p = 2
q = 1
n_paras = 3 + p + q + (m+1)

#%% Write prior and likelihood function for normal (unknown mean, known variance)
def myprior(cube):
    params = cube.copy()

    #Delta
    params[0] = cube[0] * 200

    #sig2
    params[1] = cube[1] * 30 - 15

    #microlensing
    params[2:(2+(m+1))] = [cube[i] * 20 - 10 for i in range(2, (2+(m+1)))]

    #AR paras
    #[a_0, a_1] is equivalent to polynomial z^2 + a_0 z + a_1
    params[(2+(m+1)): (2+(m+1))+p] = [cube[i] * 16 - 8 for i in range((2+(m+1)), (2+(m+1))+p)]

    #MA paras
    params[(2+(m+1))+p: (2+(m+1))+p+q] = [cube[i] * 16 - 10 for i in range((2+(m+1))+p, (2+(m+1))+p+q)]
    return params

#%% Polynomial regression function
def poly_reg(theta,t):
    '''
    Function to compute microlensing term form coefficients and time measurements
    '''
    k = len(theta)
    poly_reg = 0
    for i in range(k):
        poly_reg += theta[i]*(t**i)
    return poly_reg

#%% Parametrization function
def get_paras(para):
    p = len(para)
    roots = []
    for i in range(int(p/2)):
        coefs_ = [1, para[2*i], para[2*i+1]]
        roots.append(np.roots(coefs_)[0])
        roots.append(np.roots(coefs_)[1])
    if (p%2) == 1:
        roots.append(-1*para[p-1])
    return(np.array(np.poly(roots)))

# %% Function for centroids etc
def lorentzian(ar_paras):
    roots = np.roots(ar_paras)
    lorentz_cent = np.abs(roots.imag) / (2.0 * np.pi)
    lorentz_width = - 1.0  * roots.real / (2.0 * np.pi)
    return lorentz_cent, lorentz_width

#%% Function to compute roots from parametrized polynomial
def get_ar_roots(a_vec):
    p = len(a_vec)
    roots = []
    for i in range(int(p/2)):
        quad2 = a_vec[2*i]
        quad1 = a_vec[2*i + 1]
        poly = [1, quad2, quad1]
        roots.append(np.roots(poly)[0])
        roots.append(np.roots(poly)[1])
    if p%2 == 1:
        roots.append(-1.0 * a_vec[p-1])
    roots = np.array(roots)
    roots[np.abs(roots.imag) < 1e-10] = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots.real.argsort()]    
    return roots

#%% Function to check ordering of roots
def order_roots(roots):
    img_roots = roots.imag
    p = len(img_roots)
    order_roots = True
    idx = [i-1 for i in range(p+1) if i%2 == 1]
    img_idx = img_roots[idx]
    for i in range(img_idx.size-1):
        if np.abs(img_idx[i+1]) - np.abs(img_idx[i]) > 1e-8:
               order_roots = False
    return order_roots

#%% Log-Likelihood function
def carma_loglik(params, m = m, p = p, q = q):

    '''
    Compute log-likelihood of Time Delay model under the assumption of
    a CARMA model for the lightcurve
    '''

    loglik = 0

    #Extract parameters from input vector
    Delta = params[0]
    sigma = np.exp(params[1])
    micro = params[2:(2+(m+1))]
    a_vec = np.exp(params[(2+(m+1)): (2+(m+1))+p])
    b_vec = np.exp(params[(2+(m+1))+p: (2+(m+1))+p+q])

    ar_coefs = get_paras(a_vec)
    ma_coefs = np.array(get_paras(b_vec))
    # if ma_coefs == 1.0:
    #     ma_coefs = np.array([1.0])
    ar_roots = get_ar_roots(a_vec)

    if len(ma_coefs) < p:
        # add extra zeros to end of ma_coefs
        q = len(ma_coefs)
        ma_coefs = np.resize(np.array(ma_coefs), len(ar_roots))
        ma_coefs[q:] = 0.0

    # make sure the roots are unique
    tol = 1e-8
    for i in range(p):
        for j in range(i+1,p):
            diff_roots  = np.abs(ar_roots[i] - ar_roots[j]) / np.abs(ar_roots[i] + ar_roots[j])
    
    if np.any(diff_roots < tol):
        return -1e99
    else:

    # Setup the matrix of Eigenvectors for the Kalman Filter transition matrix. This allows us to transform
    # quantities into the rotated state basis, which makes the computations for the Kalman filter easier and faster.
    # This matrix is denoted as U in Kelly (2014), see Equation (22).
        EigenMat = np.ones((p, p), dtype=complex)
        EigenMat[1, :] = ar_roots
        for k in range(2, p):
            EigenMat[k, :] = ar_roots ** k
        # Input vector under the original state space representation [vector e in Kelly (2014) - defined between eq. (9) and (10)]
        Rvector = np.zeros(p, dtype=complex)
        Rvector[-1] = 1.0 #sigma is "added" in the StateVar calculation
        # Input vector under rotated state space representation [defined between eq. (A2) and (A3)]
        Jvector = scipy.linalg.solve(EigenMat, Rvector)  # J = inv(E) * R
        # Compute the vector of moving average coefficients in the rotated state.
        rotated_MA_coefs = ma_coefs.dot(EigenMat)
        # Calculate the stationary covariance matrix of the state vector [eq. (A3)]
        StateVar = np.empty((p, p), dtype=complex)
        for j in range(p):
            StateVar[:, j] = -(sigma ** 2) * Jvector * np.conjugate(Jvector[j]) / (ar_roots + np.conjugate(ar_roots[j]))

        # Initialize variance in one-step prediction error and the state vector
        PredictionVar = StateVar.copy()
        StateVector = np.zeros(p, dtype=complex)

        # Convert the current state to matrices for convenience, since we'll be doing some Linear algebra.
        StateVector = np.matrix(StateVector).T
        StateVar = np.matrix(StateVar)
        PredictionVar = np.matrix(PredictionVar)
        rotated_MA_coefs = np.matrix(rotated_MA_coefs)  # this is a row vector, so no transpose
        StateTransition = np.zeros_like(StateVector)
        KalmanGain = np.zeros_like(StateVector)

        #Perform centered QR parametrization
        m = len(micro) - 1

        mat = np.empty((len(t),m))
        for i in range(m):
            mat[:,i] = (t - Delta)**(i+1) - np.mean((t - Delta)**(i+1))

        #Perform QR parametrization of covariate matrix
        q = np.linalg.qr(mat)[0]


        y_centered = y
        z_centered = z - (micro[0] + np.dot(q, micro[1:]))

        #Construct composite time series 
        t_del = t - Delta
        idx = np.argsort(np.concatenate([t_del, t]))
        ordered_times = np.sort(np.concatenate([t_del, t]))
        yzcomb_centered = np.concatenate([z_centered,y_centered])[idx]
        err_comb = np.concatenate([err_B, err_A])[idx]

        kalman_mean = 0.0
        kalman_var = np.real((rotated_MA_coefs * PredictionVar * rotated_MA_coefs.H).item()) + err_comb[0]
        # Initialize the innovations, i.e., the KF residuals
        innovation = yzcomb_centered[0]


        loglik += - 0.5*(np.log(kalman_var) + np.log(2*np.pi)) - ((0.5*(yzcomb_centered[0]-kalman_mean) ** 2) / kalman_var)

        for i in range(1, ordered_times.size):
            # First compute the Kalman gain [eq. (A13)]
            KalmanGain = PredictionVar * rotated_MA_coefs.H / kalman_var
            # update the state vector [eq. (A14)]
            StateVector += innovation * KalmanGain
            # update the state one-step prediction error variance [eq. (A15)]
            PredictionVar -= kalman_var * (KalmanGain * KalmanGain.H)
            # predict the next state, do element-wise multiplication [eq. (A9)]
            dt = ordered_times[i] - ordered_times[i - 1]
            StateTransition = np.matrix(np.exp(ar_roots * dt)).T #defined right after eq. 25
            StateVector = np.multiply(StateVector, StateTransition)
            # update the predicted state covariance matrix [eq. (A10)]
            PredictionVar = np.multiply(StateTransition * StateTransition.H, PredictionVar - StateVar) + StateVar
            # now predict the observation and its variance [eq. (A11) & (A12)]
            kalman_mean = np.real((rotated_MA_coefs * StateVector).item())
            kalman_var = np.real((rotated_MA_coefs * PredictionVar * rotated_MA_coefs.H).item()) + err_comb[i]
            # Compute loglikehood
            loglik += - 0.5*(np.log(kalman_var) + np.log(2*np.pi)) - ((0.5*(yzcomb_centered[i]-kalman_mean) ** 2) / kalman_var)
            # finally, update the innovation
            innovation = yzcomb_centered[i] - kalman_mean
    return loglik

#%% Function to create array of names for parameters
def parameter_names(m,p,q):
    para_names  = ['Delta', 'sigma', 'intercept']
    for i in range(m):
        para_names.append("theta"+str(i+1))
    for j in range(p):
        para_names.append("a"+str(j+1))
    for k in range(q):
        para_names.append("b"+str(k+1))
    return para_names

#%% Initialize MultiNest
parameters = parameter_names(m,p,q)
n_params = len(parameters)
# name of the output files]
save_dir = './nest_output/'
prefix = 'tdcarma_'+str(data_file[:-4])+'_c'+str(p)+str(q)+'m'+str(m)+'_'


start = time.time()
#%%run MultiNest
result = solve(LogLikelihood=carma_loglik, Prior=myprior, n_live_points = 1000, importance_nested_sampling = True, multimodal = False, 
    use_MPI = True, n_dims=n_params, verbose=True,outputfiles_basename=save_dir+prefix, resume=True)


stop = time.time()
print('Elapsed :', (stop-start)/60, ' min')

#%% Print Bayesian evidence and parameter estimates
print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

#%% Save results
name_file = prefix + '.pkl'
open_file = open(name_file, "wb")
pickle.dump(result, open_file)
open_file.close()