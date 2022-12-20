from scipy.fft import fft, ifft, fft2, ifft2
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
os.system("rm -rf FFT_figs")
os.system("mkdir FFT_figs")

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

    f = fftfreq(len(t), np.diff(t)[0])
    x1_FFT = fft(x1)/N
    x2_FFT = fft(x2)/N
    x3_FFT = fft(x3)/N

    ####################################################################################################

    #f = f*0.108/0.89

    x = list(f[:N//2])
    y = list(np.abs(x2_FFT[:N//2])**2)

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

    plt.plot(f[:N//2], np.abs(x1_FFT[:N//2])**2, 'r-', label="uu")
    plt.plot(f[:N//2], np.abs(x2_FFT[:N//2])**2, 'g-', label="vv")
    plt.plot(f[:N//2], np.abs(x3_FFT[:N//2])**2, 'b-.', label="ww")
    plt.plot(np.exp([xmin, xmax]), np.exp([y1, y2])*0.5*10**2, 'k')
    plt.xlabel('$f$')
    plt.ylabel('PSD', fontsize=20)
    plt.xlim([0.09, 15])
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    if "turb" in line:
        plt.title(line.strip("_U_UPrime2Mean_turbulenceProperties_R.csv"))
    else:
        plt.title(line.strip("_U_UPrime2Mean_R.csv"))
    plt.savefig("FFT_figs/" +
                line.strip(".csv")+'FFT', dpi=300,  bbox_inches='tight')
    plt.show()
