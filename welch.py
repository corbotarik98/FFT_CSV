from scipy.fft import fft, ifft, fft2, ifft2

from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.fft import fftfreq
import numpy as np
import os
import math
import scipy as sp
import pandas as pd
import sympy as smp
from matplotlib import pyplot as plt
import matplotlib
font = {'family': 'sans-serif',
        'weight': 'medium',
        # 'style'  : 'italic',
        'size': 18}

matplotlib.rc('font', **font)
os.system("rm -rf WELCH_figs")
os.system("mkdir WELCH_figs")

os.system('rm nesto')
os.system('ls csvs > nesto')

f = open("nesto", "r")
nesto_ = f.readlines()

nesto = []

for line in nesto_:
    nesto.append(line.strip('\n'))


linesi = nesto

os.system("rm nesto")

lines = []

for i in linesi:
    if "fig19" or "fig17" in i:
        lines.append(i)


'''lines = ["line10_fig19_a_U.xy",
         "line11_fig19_b_U.xy",
         "line12_fig19_c_U.xy",
         "line13_fig19_d_U.xy",
         "line6_fig17_a_U.xy",
         "line7_fig17_b_U.xy"]'''

for line in lines:

    #df = pd.read_csv("csvs/"+line.strip('.xy')+".csv")
    df = pd.read_csv("csvs/"+line)
    t = list(df["time"])

    for i in range(len(t)):
        t[i] = round(float(t[i]), 8)
    T = t[-1] - t[0]  # seconds
    N = len(df["time"])  # measurements

    dt = np.diff(t)[0]
    print(dt)

    #x1 = list(df["uu"])+list(df["Ruu"])
    #x2 = list(df["vv"])+list(df["Rvv"])
    #x3 = list(df["ww"])+list(df["Rww"])

    x1 = list(df["uu"])
    x2 = list(df["vv"])
    x3 = list(df["ww"])

    '''
    for i in range(len(x1)):
        x1[i] = x1[i]-0.5

    for i in range(len(x2)):
        x2[i] = x2[i]-0.5

    for i in range(len(x3)):
        x3[i] = x3[i]-0.5'''

    nn = 18

    f1, x1_FFT = welch(x1, fs=1.0/dt, nperseg=len(x1)//nn)
    f2, x2_FFT = welch(x2, fs=1.0/dt, nperseg=len(x1)//nn)
    f3, x3_FFT = welch(x3, fs=1.0/dt, nperseg=len(x1)//nn)

    ####################################################################################################

    #f = f*0.108/0.89

    x = list(f1[:N//2])
    y = list(x1_FFT[:N//2])

    x.pop(0)
    y.pop(0)
    ymin, ymax = np.log([min(y), max(y)])
    ymid = (ymin + ymax) / 2
    xmin, xmax = np.log([min(x), max(x)])
    xmid = (xmin + xmax) / 2

    slope = - 5 / 3
    y1 = slope * (xmin - xmid) + ymid
    y2 = slope * (xmax - xmid) + ymid

    # print(f)

    plt.plot(f1, x1_FFT, 'r-', label="uu")
    plt.plot(f2, x2_FFT, 'g-', label="vv")
    plt.plot(f3, x3_FFT, 'b-.', label="ww")
    plt.plot(np.exp([xmin, xmax]), np.exp([y1, y2])*0.5*10**2, 'k')
    plt.xlabel('$f$')
    plt.ylabel('PSD', fontsize=20)
    #plt.xlim([0.09, 15])
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    if "turb" in line:
        plt.title(line.strip("_U_UPrime2Mean_turbulenceProperties_R.csv"))
    else:
        plt.title(line.strip("_U_UPrime2Mean_R.csv"))
    plt.savefig("WELCH_figs/" +
                line.strip(".csv")+'WELCH', dpi=300,  bbox_inches='tight')
    plt.show()
