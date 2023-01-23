'''
Created on 16 Dec 2022

@author: Andreagiovanni Reina.
IRIDIA, Universite' Libre de Bruxelles, Belgium
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ResultPath = "../results/set1/"
NetSizes = [100]
NetType = "erdos-renyi"
NetParams = np.arange(0.1,0.7+0.01,0.2)
NetParams = [round(np,1) for np in NetParams]
DigestTimes = np.arange(0,101,10)
TMax = 10000

def parseData():
    df = pd.DataFrame(columns=['N', 'dt', 'p', 't'])
    for netSize in NetSizes:
        for digestTime in DigestTimes:
            for netParam in NetParams:
                filename = ResultPath + "LIME_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-07_net-" + NetType + "_p-" + str(netParam) + "_dt-" + str(digestTime) + ".txt" 
                print(filename)
                with open(filename) as f:
                    res = f.read().splitlines()
                #res = [int(i) for i in res]
                for t in res:
                    df = df.append({'N':netSize, 'dt':digestTime, 'p':netParam, 't':int(t)}, ignore_index=True)
    return df

if __name__ == '__main__':
    df = parseData()
    
    netPar = 0.7
    # for dt in DigestTimes:
    #     plt.violinplot( df.loc[ (df['dt'] == dt) & (df['p'] == netPar) ]['t'], positions=[dt], widths=[5] )
    #     plt.xlabel("digestion time")
    #     plt.ylabel("adaptation time")
    #     print(dt)
    #     print( df.loc[ (df['dt'] == dt) & (df['p'] == netPar)  & (df['t'] == TMax) ]['t'].count() )
    
    dt = 0
    for netPar in NetParams:
        plt.violinplot( df.loc[ (df['dt'] == dt) & (df['p'] == netPar) ]['t'], positions=[netPar], widths=[0.1] )
        plt.xlabel("connectivity")
        plt.ylabel("adaptation time")
        print(dt)
        print( df.loc[ (df['dt'] == dt) & (df['p'] == netPar)  & (df['t'] == TMax) ]['t'].count() )
    
    plt.show()
    
    
    
    