'''
Created on 16 Dec 2022

@author: Andreagiovanni Reina.
IRIDIA, Universite' Libre de Bruxelles, Belgium
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os

ResultPath = "../results/setRW2/"
#Output_dir="../selected_plots"
Output_dir="../matrix_plots"
#Output_dir="plots-q07-100"
NetSizes = [100]
NetType = "erdos-renyi"
#NetType = "barabasi-albert"
#NetParams = np.arange(0.05,0.75+0.01,0.2)
#NetParams = np.arange(0.05,0.6+0.01,0.05)
NetParams = np.arange(0.025,0.05+0.001,0.01)
#NetParams = [0.035]
#NetParams = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.65)
NetParams = [round(np,3) for np in NetParams] #RW2
#NetParams = np.arange(3,11+1,2)
DigestTimes = np.arange(0,101,25)
#DigestTimes = [40,60]
TMax = 10000
qLowParams=[0.8]#, 0.8, 0.9]
#qLowParams=[0.7, 0.8, 0.9]
discoProbParams=[0.01, 0.02]
#discoProbParams=[0.01, 0.02, 0.05]
RewireProbs = np.arange(0.0,1+0.01,0.05)
RewireProbs = [round(np,2) for np in RewireProbs]

#NICE FIG RW
NetSizes = [50,100]
NetParams = np.arange(0.025,0.05+0.001,0.01)
NetParams = [0.035]
DigestTimes = np.arange(0,101,25)
TMax = 10000
qLowParams=[0.8]
discoProbParams=[0.02]
RewireProbs = np.arange(0.0,1+0.01,0.05)
RewireProbs = [round(np,2) for np in RewireProbs]

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
                        for rewireProb in RewireProbs:
                            #filename = ResultPath + "LIME_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.2f')if(NetType=="erdos-renyi")else str(netParam)) + "_dt-" + str(digestTime) + ".txt"
                            filename = ResultPath + "LIME_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.3f')if(NetType=="erdos-renyi")else str(netParam)) + "_dt-" + str(digestTime) + "_rwp-" + str(format(rewireProb, '.2f')) + ".txt"  
                            #(str(format(rewireProb, '.2f'))if(rewireProb>0)else "0.0") + ".txt"
                            print(filename)
                            with open(filename) as f:
                                res = f.read().splitlines()
                            #res = [int(i) for i in res]
                            for t in res:
                                df = df.append({'N':netSize, 'dt':digestTime, 'q':qLow, 'dp':discoProb, 'p':netParam, 'rw':rewireProb, 't':int(t)}, ignore_index=True)
    return df

def generateAllPlots():
    for discoProb in discoProbParams:
        for qLow in qLowParams:
            for netSize in NetSizes:
                for rewireProb in RewireProbs:
    
                    for netParam in NetParams:
                        succ = [0]*len(DigestTimes)
                        for i,dt in enumerate(DigestTimes):
                            plt.violinplot( df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow)  & (df['rw'] == rewireProb) ]['t'], positions=[dt], widths=[5] )
                            succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) & (df['t'] == TMax) ]['t'].count() / df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'].count()
                            
                        plt.xlabel("digestion time")
                        plt.ylabel("adaptation time")
                        plt.savefig(ResultPath + Output_dir + "/LIME_T-vs-DT_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.3f')if(NetType=="erdos-renyi")else str(netParam)) + "_rw-" + str(rewireProb) + ".pdf")  
                        plt.clf()
                        print(succ)
                        
                        plt.plot(DigestTimes,succ)
                        plt.ylim(-0.1,1.1)
                        plt.xlabel("digestion time")
                        plt.ylabel("adaptation probability")
                        plt.savefig(ResultPath + Output_dir + "/LIME_S-vs-DT_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.3f')if(NetType=="erdos-renyi")else str(netParam)) + "_rw-" + str(rewireProb) + ".pdf")
                        plt.clf()
                    
                    for dt in DigestTimes:
                        break
                        succ = [0]*len(NetParams)
                        for i,netParam in enumerate(NetParams):
                            plt.violinplot( df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'], positions=[netParam], widths=[0.1] )
                            succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) & (df['t'] == TMax) ]['t'].count() / df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'].count()
                
                        plt.xlabel("connectivity")
                        plt.ylabel("adaptation time")
                        plt.savefig(ResultPath + Output_dir + "/LIME_T-vs-NP_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_dt-" + str(dt) + "_rw-" + str(rewireProb) + ".pdf")  
                        plt.clf()
                        print(succ)
                        
                        plt.plot(NetParams,succ)
                        plt.ylim(-0.1,1.1)
                        plt.xlabel("connectivity")
                        plt.ylabel("adaptation probability")
                        plt.savefig(ResultPath + Output_dir + "/LIME_S-vs-NP_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_dt-" + str(dt) + "_rw-" + str(rewireProb) + ".pdf")
                        plt.clf()
                
                for dt in DigestTimes:
                    for netParam in NetParams:
                        succ = [0]*len(RewireProbs)
                        for i,rewireProb in enumerate(RewireProbs):
                            plt.violinplot( df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'], positions=[rewireProb], widths=[0.1] )
                            succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb) & (df['q'] == qLow) & (df['rw'] == rewireProb) & (df['t'] == TMax) ]['t'].count() / df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'].count()
                
                        plt.xlabel("rewiring probability")
                        plt.ylabel("adaptation time")
                        plt.savefig(ResultPath + Output_dir + "/LIME_T-vs-RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.3f')if(NetType=="erdos-renyi")else str(netParam)) + "_dt-" + str(dt) + ".pdf")  
                        plt.clf()
                        print(succ)
                        
                        plt.plot(RewireProbs,succ)
                        plt.ylim(-0.1,1.1)
                        plt.xlabel("rewiring probability")
                        plt.ylabel("adaptation probability")
                        plt.savefig(ResultPath + Output_dir + "/LIME_S-vs-RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + (format(netParam, '.3f')if(NetType=="erdos-renyi")else str(netParam)) + "_dt-" + str(dt) + ".pdf")
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
    plt.savefig(ResultPath + Output_dir + "/niceFig_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + ".pdf")


def plotNiceFigureRW():
    discoProb=discoProbParams[0]
    qLow=qLowParams[0]
    netParam=NetParams[0]
    # DigestTimes=np.arange(0,101,25)
    #netSize=50
    #for discoProb in discoProbParams:
    #for qLow in qLowParams:
        #for rewireProb in RewireProbs:
    plt.clf()
    plt.axhline( y=0, color='grey', linestyle=':',linewidth=0.5)
    plt.axhline( y=1, color='grey', linestyle=':',linewidth=0.5)    
    colors = [(0, 0, 0), (0.239, 0.718, 0.914), (0.969, 0.282, 0.647), (0.208, 0.608, 0.451), (0.898, 0.624, 0), (0.133, 0.443, 0.698), (0.941, 0.894, 0.258), (0.835, 0.369, 0)]    
    for netSize in NetSizes:
        col_i=-1
        for dt in DigestTimes:
            col_i+=1
            #for netParam in NetParams:
            succ = [0]*len(RewireProbs)
            stderr1 = [0]*len(RewireProbs)
            stderr2 = [0]*len(RewireProbs)
            for i,rewireProb in enumerate(RewireProbs):            
                total_runs = df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['rw'] == rewireProb) ]['t'].count()
                succ[i] = 1 - df.loc[ (df['N'] == netSize) & (df['dt'] == dt) & (df['p'] == netParam) & (df['dp'] == discoProb)  & (df['q'] == qLow) & (df['t'] == TMax) & (df['rw'] == rewireProb) ]['t'].count() / total_runs
                stderr1[i] = succ[i] + 1.96 * np.sqrt(succ[i] * (1-succ[i]) / total_runs) 
                stderr2[i] = succ[i] - 1.96 * np.sqrt(succ[i] * (1-succ[i]) / total_runs)
            plt.plot(RewireProbs,succ,"--"if not netSize==NetSizes[0] else "-",color=colors[col_i])
            plt.fill_between(RewireProbs,stderr1,stderr2, alpha=0.2,color=colors[col_i])
    #plt.ylim(-0.1,1.1)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.xlabel("rewiring probability")
    plt.ylabel("probability population adapts to better option")
    
    
    # Creating legend handles for Deliberation time
    dtmarkers = [0]*len(DigestTimes)
    for i,dt in enumerate(DigestTimes):
        dtmarkers[i] = mlines.Line2D([], [], color=colors[i], linestyle='-', linewidth=2, label=str(dt))
    # Creating legend handles for Group size
    nsmarkers = [mlines.Line2D([], [], color='grey', linestyle="--" if not ns == NetSizes[0] else "-", linewidth=2, label=str(ns)) for ns in NetSizes]
    # Creating legends
    legend1 = plt.legend(handles=dtmarkers, bbox_to_anchor=(0.35, 1.15), loc='upper center', fancybox=True, ncol=len(dtmarkers), handlelength=0.8 )
    legend1.set_title('Deliberation time')
    legend2 = plt.legend(handles=nsmarkers, bbox_to_anchor=(0.85, 1.15), loc='upper center', fancybox=True, ncol=len(NetSizes), handlelength=1.5 )
    legend2.set_title('Group size')
    # Adding both legends to the plot
    plt.gca().add_artist(legend1)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    #plt.show()
    print(ResultPath + Output_dir + "/niceFig_RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(format(netParam, '.3f')) +  ".pdf")
    plt.savefig(ResultPath + Output_dir + "/niceFig_RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(format(netParam, '.3f')) +  ".pdf", bbox_inches='tight')
    #plt.clf()
    
def plotMatrixRW2(df, netSize, discoProb, qLow, netParam):
    sub = df.loc[
        (df["N"] == netSize) &
        (df["p"] == netParam) &
        (df["dp"] == discoProb) &
        (df["q"] == qLow)
    ].copy()

    if sub.empty:
        print("No data after filtering for N={}, q={}, dp={}, p={}".format(
            netSize, qLow, discoProb, netParam
        ))
        return

    # Axis values
    rw_vals = np.sort(sub["rw"].unique())
    dt_vals = np.sort(sub["dt"].unique())

    # Compute success (t < TMax)
    sub["ok"] = sub["t"] < TMax
    Z_df = sub.groupby(["dt", "rw"])["ok"].mean().unstack("rw")

    # Ensure full grid
    Z_df = Z_df.reindex(index=dt_vals, columns=rw_vals)
    Z = Z_df.values

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="equal",
        cmap="gnuplot",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_xlabel("rewiring probability")
    ax.set_ylabel("digestion time")
    ax.set_title("N={}, q={}, dp={}, p={}, t < {}".format(
        netSize, qLow, discoProb, netParam, TMax
    ))

    ax.set_xticks(np.arange(len(rw_vals)))
    ax.set_xticklabels(["{:.2f}".format(x) for x in rw_vals], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(dt_vals)))
    ax.set_yticklabels(dt_vals)

    # --- Top colorbar (old Matplotlib safe) ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.15)
    
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("adaptation probability")
    cax.xaxis.set_label_position("top")
    cax.xaxis.tick_top()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    #plt.show()
    print(ResultPath + Output_dir + "/matrix_RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(format(netParam, '.3f')) +  ".pdf")
    plt.savefig(ResultPath + Output_dir + "/matrix_RW_N-" + str(netSize) + "_mt-" + str(TMax) + "_q-" + str(qLow) +  "_dp-" + str(discoProb) + "_net-" + NetType + "_p-" + str(format(netParam, '.3f')) +  ".pdf", bbox_inches='tight')


def plotMatrixRW(df, netSize, discoProb, qLow, netParam):
    sub = df.loc[
        (df["N"] == netSize) &
        (df["p"] == netParam) &
        (df["dp"] == discoProb) &
        (df["q"] == qLow)
    ].copy()

    if sub.empty:
        print("No data after filtering for N={}, q={}, dp={}, p={}".format(
            netSize, qLow, discoProb, netParam
        ))
        return

    rw_vals = np.sort(sub["rw"].unique())
    dt_vals = np.sort(sub["dt"].unique())

    sub["ok"] = sub["t"] < TMax
    Z_df = sub.groupby(["dt", "rw"])["ok"].mean().unstack("rw")
    Z_df = Z_df.reindex(index=dt_vals, columns=rw_vals)
    Z = Z_df.values

    # --- Make cells square by choosing figsize proportional to grid ---
    cell = 0.35  # inches per cell; tweak to taste
    fig_w = max(6.0, len(rw_vals) * cell)
    fig_h = max(3.5, len(dt_vals) * cell + 0.9)  # extra room for top cbar

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.12, 1.0], hspace=0.15)

    cax = fig.add_subplot(gs[0])   # top colourbar axis
    ax  = fig.add_subplot(gs[1])   # main heatmap axis

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="equal",
        cmap="gnuplot",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_xlabel("rewiring probability")
    ax.set_ylabel("digestion time")
    ax.set_title("N={}, q={}, dp={}, p={}, t < {}".format(
        netSize, qLow, discoProb, netParam, TMax
    ))

    ax.set_xticks(np.arange(len(rw_vals)))
    ax.set_xticklabels(["{:.2f}".format(x) for x in rw_vals], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(dt_vals)))
    ax.set_yticklabels(dt_vals)

    # --- Top horizontal colourbar ---
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("adaptation probability")
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position("top")
    cax.set_yticks([])  # keep the colourbar row clean

    # --- Save ---
    outpath = (ResultPath + Output_dir +
               "/matrix_RW_N-{}_mt-{}_q-{}_dp-{}_net-{}_p-{}.pdf".format(
                   netSize, TMax, qLow, discoProb, NetType, format(netParam, ".3f")
               ))
    fig.savefig(outpath, bbox_inches="tight")
    print(outpath)

    plt.close(fig)
    
if __name__ == '__main__':
    if not os.path.isdir(ResultPath + Output_dir + "/"):
        os.makedirs(ResultPath + Output_dir + "/")
        print(ResultPath + Output_dir + "/")
    
    qs=[0.7,0.8,0.9]
    dps=[0.01,0.02]
    NetSizes=[100]
    for qi in qs:
        for dp in dps:
            qLowParams=[qi]
            discoProbParams=[dp]
            df = parseData()
            #generateAllPlots()
            #plotNiceFigure()
            #plotNiceFigureRW()  
            plotMatrixRW(df=df, netSize=NetSizes[0], discoProb=discoProbParams[0], qLow=qLowParams[0], netParam=NetParams[0])
    