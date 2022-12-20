import pandas as pd
import os
import numpy as np

os.system('rm -rf csvs')
os.system('mkdir csvs')

time = []

location = 2.25
deltaT = 0.005


os.system('rm nesto')
os.system('ls combined > nesto')

f = open("nesto", "r")
nesto_ = f.readlines()

nesto = []

for line in nesto_:
    nesto.append(line.strip('\n'))


lines = nesto

os.system("rm nesto")

'''

try:
    f = open("times", "r")
    time_ = f.readlines()

    time = []

    for line in time_:
        time.append(line.strip('\n'))

except:
    time = np.arange(0, 540.005, 0.005)

'''


for line in lines:
    f = open('combined/'+line, "r")
    data = f.readlines()
    time = []
    u = []
    v = []
    w = []

    uu = []
    uv = []
    uw = []
    vv = []
    vw = []
    ww = []

    Ruu = []
    Ruv = []
    Ruw = []
    Rvv = []
    Rvw = []
    Rww = []

    for d in data:
        if str(location) in d:
            d.strip("\n")
            a = d.split()
            time.append(a[1])
            u.append(a[2])
            v.append(a[3])
            w.append(a[4])

            uu.append(a[5])
            uv.append(a[6])
            uw.append(a[7])
            vv.append(a[8])
            vw.append(a[9])
            ww.append(a[10])
            Ruu.append(a[11])
            Ruv.append(a[12])
            Ruw.append(a[13])
            Rvv.append(a[14])
            Rvw.append(a[15])
            Rww.append(a[16])

        for i in range(len(time)):
            dt = np.diff(t)[1]
            if dt != deltaT:
                print("ERROR DELTA t")
                break

    df = pd.DataFrame(list(zip(time, u, v, w, uu, uv, uw, vv, vw, ww, Ruu, Ruv, Ruw, Rvv, Rvw, Rww)),
                      columns=['time', 'u', 'v', 'w', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'Ruu', 'Ruv', 'Ruw', 'Rvv', 'Rvw', 'Rww'])
    df.to_csv('csvs/'+line.strip('.xy')+'.csv', index=False)
