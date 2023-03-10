'''
Created on 16 Dec 2022

@author: Andreagiovanni Reina.
IRIDIA, Universite' Libre de Bruxelles, Belgium
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ResultPath = "../results/set4/"
NetSizes = [1000]
#NetType = "erdos-renyi"
NetType = "barabasi-albert"
#NetParams = np.arange(0.05,0.75+0.01,0.2)
#NetParams = np.arange(0.05,0.6+0.01,0.05)
#NetParams = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.65)
#NetParams = [round(np,2) for np in NetParams]
NetParams = np.arange(3,11+1,2)
DigestTimes = np.arange(0,1001,200)
#DigestTimes = [40,60]
TMax = 100000
qLowParams=[0.7, 0.8, 0.9]
discoProbParams=[0.01, 0.02, 0.05]

#NetParams = [0.7]
#qLowParams=[0.9]
#discoProbParams=[0.02]

def parseData():
    df = pd.DataFrame(columns=['N', 'dt', 'q', 'dp', 'p', 't'])
    for discoProb in discoProbParams:
        for qLow in qLowParams:
            for netSize in NetSizes:
                for digestTime in DigestTimes:
                    for netParam in NetParams:
                        filename = ResultPath + "LIME_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.2f')if(NetType=="erdos-renyi")else str(netParam)) + "_dt-" + str(digestTime) + ".txt"
                        print(filename)
                        with open(filename) as f:
                            res = f.read().splitlines()
                        #res = [int(i) for i in res]
                        for t in res:
                            df = df.append({'N':netSize, 'dt':digestTime, 'q':qLow, 'dp':discoProb, 'p':netParam, 't':int(t)}, ignore_index=True)
    return df

def generateAllPlots():
    for discoProb in discoProbParams:
        for qLow in qLowParams:
            for netSize in NetSizes:
    
                for netParam in NetParams:
                    succ = [0]*len(DigestTimes)
                    for i,dt in enumerate(DigestTimes):
                        plt.violinplot( df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) ]['t'], positions=[dt], widths=[5] )
                        succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['t'] == TMax) ]['t'].count() / df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) ]['t'].count()
                        
                    plt.xlabel("digestion time")
                    plt.ylabel("adaptation time")
                    plt.savefig(ResultPath + "plots/LIME_T-vs-DT_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(netParam) + ".pdf")  
                    plt.clf()
                    print(succ)
                    
                    plt.plot(DigestTimes,succ)
                    plt.ylim(-0.1,1.1)
                    plt.xlabel("digestion time")
                    plt.ylabel("adaptation probability")
                    plt.savefig(ResultPath + "plots/LIME_S-vs-DT_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(netParam) + ".pdf")
                    plt.clf()
                
                for dt in DigestTimes:
                    succ = [0]*len(NetParams)
                    for i,netParam in enumerate(NetParams):
                        plt.violinplot( df.loc[ (df['dt'] == dt) & (df['p'] == netParam) ]['t'], positions=[netParam], widths=[0.1] )
                        succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['t'] == TMax) ]['t'].count() / df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) ]['t'].count()
            
                    plt.xlabel("connectivity")
                    plt.ylabel("adaptation time")
                    plt.savefig(ResultPath + "plots/LIME_T-vs-NP_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_dt-" + str(dt) + ".pdf")  
                    plt.clf()
                    print(succ)
                    
                    plt.plot(NetParams,succ)
                    plt.ylim(-0.1,1.1)
                    plt.xlabel("connectivity")
                    plt.ylabel("adaptation probability")
                    plt.savefig(ResultPath + "plots/LIME_S-vs-NP_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_dt-" + str(dt) + ".pdf")
                    plt.clf()
    
    #plt.show()

def plotNiceFigure():
    discoProb=0.02
    qLow=0.9
    netSize=50
    plt.clf()
    plt.axhline( y=0, color='grey', linestyle=':',linewidth=0.5)
    plt.axhline( y=1, color='grey', linestyle=':',linewidth=0.5)
    for dt in DigestTimes:
        succ = [0]*len(NetParams)
        stderr1 = [0]*len(NetParams)
        stderr2 = [0]*len(NetParams)
        for i,netParam in enumerate(NetParams):
            total_runs = df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) ]['t'].count()
            succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['t'] == TMax) ]['t'].count() / total_runs
            stderr1[i] = succ[i] + 1.96 * np.sqrt(succ[i] * (1-succ[i]) / total_runs) 
            stderr2[i] = succ[i] - 1.96 * np.sqrt(succ[i] * (1-succ[i]) / total_runs) 
                
        plt.plot(NetParams,succ,linewidth=2, label="t="+str(dt))
        plt.fill_between(NetParams,stderr1,stderr2, alpha=0.2)
    plt.xlabel("network connectivity")
    plt.ylabel("probability population adapts to better option")
    plt.ylim(-0.02,1.02)
    plt.legend(title="Deliberation time", loc=(0.73,0.47))
    plt.savefig(ResultPath + "plots/niceFig_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + ".pdf")
    
    
if __name__ == '__main__':
    df = parseData()
    
    if not os.path.isdir(ResultPath + "plots/"):
        os.makedirs(ResultPath + "plots/")
        
    generateAllPlots()
    #plotNiceFigure()
    
    