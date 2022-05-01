# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

datadir = './PrecisionRecall'

P_matrix     = np.load(datadir + '/precision_matrix.npy')
R_matrix     = np.load(datadir + '/recall_matrix.npy') 

tol_max = 6         # Number of precision-recall curves to plot

plt.figure(figsize=(3.2,2.0), dpi=150)
plt.plot(np.transpose(R_matrix[0:tol_max,:]),
         np.transpose(P_matrix[0:tol_max]))

# Plot F1 isolines:
for F1 in np.arange(0.1,1,0.05):
    X = np.arange(F1/2+0.002,1.012,0.01)
    Y = X*F1/(2*X-F1)
    plt.plot(X,Y,'-.',color='gray',linewidth=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.grid()
plt.xlabel('Recall',fontsize=8,family='arial')
plt.ylabel('Precision',fontsize=8,family='arial')
plt.title('Average precision-Recall curve of model HS_01' +
          '\napplied to the test set',fontsize=8,family='arial')
legend_str = [str(tol) for tol in range(0,tol_max)]
legend_str.append('F1 isolines')
plt.legend(legend_str, title = 'tolerance:', prop = {'size':8})

ax = plt.gca()
ax.set_axisbelow(True)
ax.tick_params(axis='both', which='major', labelsize=6)
#plt.tight_layout()
plt.savefig('PrecisionRecall.svg')