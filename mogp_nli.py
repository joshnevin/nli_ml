import numpy as np
# from projectile import simulator_multioutput, print_results
import mogp_emulator
from mogp_emulator.MeanFunction import Coefficient, LinearMean, MeanFunction
import pandas as pd
from scipy.io import savemat, loadmat
try:
    import matplotlib.pyplot as plt
    makeplots = True
except ImportError:
    makeplots = False
import pickle
import time
launch_powers = loadmat("data/input_data_mW.mat")['lnch_pow_mW']
noise_powers = loadmat("data/output_data_mW.mat")['tot_noise_pow_mW']

num_channels = launch_powers.shape[1]
num_examples = launch_powers.shape[0]
num_train = 150
num_test = 100

# kernel_function = 'Matern52'
kernel_function = 'SquaredExponential'

nugget_type = "fit"
mf = "I(x[0]**3) + I(x[1]**3) + I(x[2]**3) + I(x[3]**3) + I(x[4]**3)"


noise_powers_shaped = np.zeros([num_channels,num_examples])
for i in range(num_channels):
    noise_powers_shaped[i] = noise_powers[:,i]

if __name__ == "__main__": # this is required for multiprocessing to work correctly!

# Next, fit the surrogate MOGP model using MAP with the default priors

    gp = mogp_emulator.MultiOutputGP(launch_powers[:150], noise_powers_shaped[:,:150], mean=mf,
    nugget = nugget_type, kernel = kernel_function)
    start = time.time()
    gp = mogp_emulator.fit_GP_MAP(gp, n_tries=2)
    end = time.time()
    print(end-start)
    print(gp.emulators[0].theta)
    print(gp.emulators[1].theta)
    print(gp.emulators[2].theta)
    print(gp.emulators[3].theta)
    print(gp.emulators[4].theta)
    pickle.dump(gp, open("results/gp_150"+"_"+kernel_function+"_"+nugget_type+"_mf.pkl", 'wb'))
    pickle.dump(end-start, open("results/gp_150"+"_"+kernel_function+"_"+nugget_type+"_mf_traintime.pkl", 'wb'))

# Validate emulator by comparing to true simulated value
# To compare with the emulator, use the predict method to get mean and variance
# values for the emulator predictions and see how many are within 2 standard
# deviations
    start = time.time()
    predictions = gp.predict(launch_powers[150:])
    end = time.time()
    print(end-start)
    pickle.dump(predictions, open("results/predictions_100_"+kernel_function+"_"+nugget_type+"_mf.pkl", 'wb'))
    pickle.dump(end-start, open("results/gp_150"+"_"+kernel_function+"_"+nugget_type+"_mf_testtime.pkl", 'wb'))
