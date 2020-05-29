import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
#2001,    ##2004,2009,2014,2019
# BJP 2004 34.77 2009 41.63 2014 43.00% 2019 51.38
# INC 2004 36.82 2009 37.65 2014 40.80% 2019 31.88
# GDP = 166747,337559,913923,1588303
# growth_rate =  2.2,21.7,11.80,6.8
# unemploy_rate =3.10,2.48,2.76,2.55
# x = pd.Series(range(10, 20))
BJP = np.array([[34.77,41.63,43.00,51.38],
                [166747,337559,913923,1588303],
                [2.2,21.7,11.80,6.8],
                [3.10,2.48,2.76,2.55]])

INC = np.array([[36.82,37.65,40.80,31.88],
                [166747,337559,913923,1588303],
                [2.2,21.7,11.80,6.8],
                [3.10,2.48,2.76,2.55]])

corr_matrix_BJP = np.corrcoef(BJP).round(decimals=2)
corr_matrix_INC = np.corrcoef(INC).round(decimals=2)
print("BJP: \n",corr_matrix_BJP)
print("INC: \n",corr_matrix_INC)
fig, ax = plt.subplots()
im = ax.imshow(corr_matrix_BJP)
im.set_clim(-1, 1)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1, 2, 3), ticklabels=('Vote%', 'GDP', 'Growth Rate','unemploy_rate'))
ax.yaxis.set(ticks=(0, 1, 2, 3), ticklabels=('Vote%', 'GDP', 'Growth Rate','unemploy_rate'))
for i in range(4):
    for j in range(4):
        ax.text(j, i, corr_matrix_BJP[i, j], ha='center', va='center',color='r')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
ax.set_title("BJP")
plt.savefig("./module1_output/BJPcorr.png")
plt.show()

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(corr_matrix_INC)
im2.set_clim(-1, 1)
ax2.grid(False)
ax2.xaxis.set(ticks=(0, 1, 2, 3), ticklabels=('Vote%', 'GDP', 'Growth Rate','unemploy_rate'))
ax2.yaxis.set(ticks=(0, 1, 2, 3), ticklabels=('Vote%', 'GDP', 'Growth Rate','unemploy_rate'))
#ax.set_ylim(2.5, -0.5)
for i in range(4):
    for j in range(4):
        ax2.text(j, i, corr_matrix_INC[i, j], ha='center', va='center',
                color='r')
cbar = ax2.figure.colorbar(im2, ax=ax2, format='% .2f')
ax2.set_title("INC")
plt.savefig("./module1_output/INCcorr.png")
plt.show()
