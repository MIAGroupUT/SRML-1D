import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['svg.fonttype'] = 'none'

model_list = ['HS_001', '1%', '4%', '8%', '16%', '32%', \
              'HS_001_polydisperse', 'R']
filedir = 'ModelComparison'

tol_list = np.arange(10)


fig = plt.figure(figsize=(3.5,2.3), dpi=150)
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
ax.tick_params(axis='both', which='major', labelsize=7)

fontdict = {'family': 'arial'}
ax.set_xticklabels(ax.get_xticks(), fontdict)
ax.set_yticklabels(ax.get_yticks(), fontdict)


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
ax.tick_params(axis='both', which='major', labelsize=7)

fontdict = {'family': 'arial'}
ax2.set_xticklabels(ax2.get_xticks(), fontdict)

plt.savefig('ModelComparison_noise.svg')