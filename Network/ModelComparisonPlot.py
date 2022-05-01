import matplotlib.pyplot as plt
import numpy as np


model_list = ['H', 'HS_001', 'HS_01', 'HS_1', 'S_001', 'S_01', 'S_1', 'R']
filedir = 'ModelComparison'

tol_list = np.arange(10)


fig = plt.figure(figsize=(3.2,2.0), dpi=150)
for k,modelname in enumerate(model_list):
    F1_array = np.load(filedir + '/F1_test_' + modelname + '.npy')
    plt.plot(F1_array)
    
plt.xlabel('tolerance',fontsize=8,family='arial')
plt.ylabel('F1 score',fontsize=8,family='arial')
plt.title('Maximum F1 (for optimal threshold)',fontsize=8,family='arial')
plt.legend(model_list,prop = {'size':6})
plt.grid()
plt.ylim([0,1])
plt.xlim([0,9])

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=6)


# Set up second x axis:
dt = 16             # Sampling interval (ns)
f0 = 0.0017         # Transmit frequency (GHz)
t_lambda = 2/f0     # Diffraction time (ns)

t_tol = tol_list*dt # Localisation tolerance (ns)


ax1 = fig.axes[0]
ax2 = ax1.twiny()
ax2.set_xlim([t_tol[0]/t_lambda,t_tol[-1]/t_lambda])

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=6)

plt.savefig('ModelComparison.svg')