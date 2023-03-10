'''
Created on 2 August 2022

@author: Andreagiovanni Reina.
IRIDIA, Universite' Libre de Bruxelles, Belgium
'''

import sys
import os
import configparser
from enum import Enum
import math
import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
DEFAULT_CONFIG_FILENAME="/Users/joefresna/LessIsMoreOpinionDynamics/config/bestOf2.config"

class NetworkType(Enum):
    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    RGG = 3
    RGG_FIXED_DEGREE = 4

def initNetwork(netType, numNodes, randomSeed, netParam):
    if (netType == NetworkType.FULLY_CONNECTED):
        return nx.complete_graph(numNodes) #np.repeat(0, self.numNodes)
    elif (netType == NetworkType.ERSOS_RENYI):
        graph = nx.erdos_renyi_graph(numNodes, netParam, randomSeed)
        i = 0
        while ( not nx.is_connected( graph ) ):
            if (DEBUG): print("Graph was not connected; Resampling!")
            i = i+1
            graph = nx.erdos_renyi_graph(numNodes, netParam, randomSeed*i*2211)
        return graph
    elif (netType == NetworkType.BARABASI_ALBERT):
        return nx.barabasi_albert_graph(numNodes, netParam, randomSeed)
    elif (netType == NetworkType.RGG or netType == NetworkType.RGG_FIXED_DEGREE):
        connected = False
        while (not connected):
            graph=nx.random_geometric_graph(numNodes, netParam, pos={i: np.random.uniform(size=2) for i in range(numNodes) })
            # pos_dict = {}
            # graph = nx.Graph()
            # for i in np.arange(0,numNodes):
            #     pos_dict[i] = np.array( [np.random.uniform(0, 1), np.random.uniform(0, 1)] )  
            #     graph.add_node(i)
            #     i += 1
            #
            # for agent in np.arange(0,numNodes):
            #     for neigh, neighPos in pos_dict.items():
            #         if (not neigh == agent) and (np.linalg.norm(pos_dict[agent]-neighPos) <= netParam):
            #             graph.add_edge(agent, neigh)
                
            connected = nx.is_connected( graph )
            if not connected:
                if (DEBUG): print("Graph was not connected; Resampling!")
            #nx.set_node_attributes(graph, pos_dict, "pos")
        return graph
            
if __name__ == '__main__':
    if DEBUG: 
        print("Process Started")
    if len(sys.argv)>1 :
        configFile = sys.argv[1]
    else:
        configFile = DEFAULT_CONFIG_FILENAME
    
    if not (os.path.isfile(configFile)):
        print("Impossible to open config file: ", configFile)
        sys.exit()
    
    if DEBUG:
        print("Reading properties file: " + configFile)
    # opening the config file
    config = configparser.ConfigParser()
    config.read(configFile)
   
    ### Load parameters from config file
    ## -- Experiment params
    randomSeed = config.getint('experiment', 'randomSeed')
    numberOfExperiments = config.getint('experiment', 'numberOfExperiments')
    outputTxtFile = config.get('experiment', 'outputTxtFile')
    cluster = config.getboolean('experiment', 'cluster')
    maxTime = config.getint('experiment', 'maxTime')
    optionQualities = json.loads( config['experiment']['optionQualities'] )
    discoveryProbability = config.getfloat('experiment', 'discoveryProbability')
    digestionTime = config.getint('experiment', 'digestionTime')
     
    ## -- Network params
    numOfNodes = config.getint('network', 'numberOfNodes')
    netTypeStr = config.get('network', 'netType')
    if (netTypeStr == 'full'):
        netType =NetworkType.FULLY_CONNECTED
    elif (netTypeStr == 'erdos-renyi'):
        netType = NetworkType.ERSOS_RENYI
    elif (netTypeStr == 'barabasi-albert'):
        netType = NetworkType.BARABASI_ALBERT
    elif (netTypeStr == 'rgg'):
        netType = NetworkType.RGG
    elif (netTypeStr == 'rgg-fixed-degree'):
        netType = NetworkType.RGG_FIXED_DEGREE
    else:
        print("Non valid input for parameter [Network].netType. Error on value '" + netTypeStr + "'. Valid values are: 'full', 'erdos-renyi', 'barabasi-albert', 'rgg', 'rgg-fixed-degree'")
        sys.exit()
    if (not netType == NetworkType.FULLY_CONNECTED):           
        if (netType == NetworkType.BARABASI_ALBERT):
            netParam  = config.getint('network', 'networkParameter')
            if (netParam >= numOfNodes):
                print("ERROR! With barabasi-albert networks the number of edges must be smaller than number of edges. Invalid network parameter = " + str(netParam) + ".")
                sys.exit()
        else:
            netParam = config.getfloat('network', 'networkParameter')
            if (netType == NetworkType.RGG_FIXED_DEGREE): # convert the fixed degree in the RGG's communication radius
                netParam = math.sqrt( netParam / (math.pi * numOfNodes) )
    else:
        netParam=0
    rewiringProbability = config.getfloat('network', 'rewiringProbability')
    
    ## Pring out config
    if DEBUG:
        print( "randomSeed: " + str(randomSeed) )
        print( "numberOfExperiments: " + str(numberOfExperiments) )
        print( "outputTxtFile: " + str(outputTxtFile) )
        print( "cluster: " + str(cluster) )
        print( "maxTime: " + str(maxTime) )
        print( "optionQualities: " + str(optionQualities) )
        print( "discoveryProbability : " + str(discoveryProbability) )
        print( "digestionTime : " + str(digestionTime) )
        print( "numOfNodes: " + str(numOfNodes) )
        print( "netType: " + str(netTypeStr) )
        print( "networkParameter: " + str(netParam) )
        print( "rewiringProbability : " + str(rewiringProbability) )
    
    np.random.seed(randomSeed)
    convergence_times = [maxTime]*numberOfExperiments
    for rep in range(numberOfExperiments):
        ## Creating the network
        graph = initNetwork(netType, numOfNodes, randomSeed+(rep*555), netParam)
        ## Initialising the opinions with all nodes starting from opinion 2
        opinions = { n:2 for n in range(numOfNodes) }
        digestion = { n:0 for n in range(numOfNodes) }
        #print(opinions)
        nx.set_node_attributes(graph, opinions, "opinion")
        nx.set_node_attributes(graph, digestion, "digestion")

        for t in np.arange(1,maxTime+1):
            # Select a random node
            node = np.random.randint(numOfNodes)
            
            # network re-wiring
            if np.random.uniform() < rewiringProbability:
                rewiring_attempts = 0 # temporary counter to avoid deadlocks
                #print(nx.is_connected( graph ))
                #print("Node " + str(node) + " had " + str( len(graph.edges(node)) ) + " friends.")
                while (rewiring_attempts < 1000): # try to rewire the network for a maximum num of times before giving up
                    # if rewiring_attempts > 10: print("Node " + str(node) + " has " + str( len(graph.edges(node)) ) + " friends.")
                    # select a random edge of the node
                    edge_id = np.random.randint(len(graph.edges(node)))
                    edge_first = list(graph.edges(node))[edge_id]
                    # select another random edge that does not involve node and its neighbours
                    edge_second = list(graph.edges())[ np.random.randint( graph.number_of_edges() ) ]
                    # randomly choose to join 1st element with 1st element or 1st element with 2nd element (when combining edge_first and edge_second)
                    invert_edge_order = np.random.choice([True,False])
                    if invert_edge_order:
                        # check if the rewiring would create a loop-edge from/to the same node
                        if (edge_first[0] == edge_second[0] or edge_first[1] == edge_second[1]):
                            continue
                        # create the two new edges by swapping the nodes (and placing them in ascending order)
                        edge_new_1 = (edge_first[0], edge_second[0]) if edge_first[0] < edge_second[0] else (edge_second[0], edge_first[0]) 
                        edge_new_2 = (edge_first[1], edge_second[1]) if edge_first[1] < edge_second[1] else (edge_second[1], edge_first[1]) 
                    else:
                        # check if the rewiring would create a loop-edge from/to the same node
                        if (edge_first[0] == edge_second[1] or edge_first[1] == edge_second[0]):
                            continue
                        # create the two new edges by swapping the nodes (and placing them in ascending order)
                        edge_new_1 = (edge_first[0], edge_second[1]) if edge_first[0] < edge_second[1] else (edge_second[1], edge_first[0]) 
                        edge_new_2 = (edge_first[1], edge_second[0]) if edge_first[1] < edge_second[0] else (edge_second[0], edge_first[1])
                    # if none of the two new edges already exists in the graph, then they are good, and exit the loop 
                    if (edge_new_1 not in list(graph.edges())) and (edge_new_2 not in list(graph.edges())):
                        break
                if (rewiring_attempts < 1000): # all checks have passed, thus we remove the edges and add the new ones
                    # remove the edges
                    graph.remove_edge(edge_first[0], edge_first[1])
                    graph.remove_edge(edge_second[0], edge_second[1])
                    #print("Removing edges " + str(edge_first) + " and " + str( edge_second ) + ".")
                    # add the new edges
                    graph.add_edge(edge_new_1[0], edge_new_1[1])
                    graph.add_edge(edge_new_2[0], edge_new_2[1])
                    #print("Adding edges " + str(edge_new_1) + " and " + str( edge_new_2 ) + ".")
                else:
                    print("Failed to rewire node " + str(node) + " which already has " + str( len(graph.edges(node)) ) + " friends.")
                
                # select a random node that is not itself
                new_neigh = node
                while (new_neigh == node):
                    new_neigh = np.random.randint(numOfNodes)
                # add a new edge with the new_neigh
                graph.add_edge(node, new_neigh)
                #print("Node " + str(node) + " has " + str( len(graph.edges(node)) ) + " friends.")
            
            # Update the digestion timers
            digestion = nx.get_node_attributes(graph, 'digestion')
            digestion = { n: max(0,d-1) for n,d in digestion.items() }
            nx.set_node_attributes(graph, digestion, "digestion")
            
            change = False
            # Decide whether have independent discovery or social interaction
            if np.random.uniform() < discoveryProbability: # independent discovery 
                if (graph.nodes[node]['opinion']==2): change = True
                graph.nodes[node]['opinion']=1
                # if DEBUG: print("Discovery by node " + str(node))
            else: # social interaction
                use_listening_prob = True
                if use_listening_prob:
                    ## Selecting a neighbor with probability proportional to their quality (normalised by the number of neighbours)
                    neighs = [ (n,optionQualities[ graph.nodes[n]['opinion']-1 ]) for n in graph.neighbors(node) if graph.nodes[n]['digestion']==0 ]
                    # print("Before norm " + str(neighs))
                    probSum = sum(p for _,p in neighs)
                    neighs = [ (n,p/probSum) for n,p in neighs ]
                    # print("After  norm " + str(neighs))
                    randNum = np.random.uniform()
                    # print("My opinion is " + str(graph.nodes[node]['opinion']))
                    bottom = 0
                    for neigh,prob in neighs:
                        if randNum < bottom+prob:
                            if not (graph.nodes[node]['opinion'] == graph.nodes[neigh]['opinion']):
                                graph.nodes[node]['opinion'] = graph.nodes[neigh]['opinion']
                                graph.nodes[node]['digestion'] = digestionTime
                                change = True
                            # if DEBUG: print("I'm " + str(node) + " Selected neigh " + str(neigh) + " w/ op: " + str(graph.nodes[neigh]['opinion']))
                            break 
                        else:
                            bottom += prob
                else:
                    ## Select one random neighbor from the speaking neighbors (i.e. first I determine which neighbors are speaking and then I pick one uniformly random)
                    neighs = [n for n in graph.neighbors(node) if (np.random.uniform() < optionQualities[ graph.nodes[n]['opinion']-1 ]) and (graph.nodes[n]['digestion']==0) ]
                    #print(neighs)
                    if len(neighs) > 0:
                        neigh = np.random.choice(neighs)
                        if not (graph.nodes[node]['opinion'] == graph.nodes[neigh]['opinion']):
                            graph.nodes[node]['opinion'] = graph.nodes[neigh]['opinion']
                            graph.nodes[node]['digestion'] = digestionTime
                            change = True
            
            # if (change): print(len([o for o in nx.get_node_attributes(graph, 'opinion').values() if o == 1]))

            if (False and change):                        
                opinion = nx.get_node_attributes(graph, 'opinion')
                print(opinion)
                digestion = nx.get_node_attributes(graph, 'digestion')
                print(digestion)
                ## Plotting the graph with colored nodes
                decisionColors=[]
                for n in graph.nodes():
                    if (graph.nodes[n]['opinion']==1): decisionColors.append('r') 
                    elif  (graph.nodes[n]['opinion']==2): decisionColors.append('g')
                    else: decisionColors.append('y')
                if (netType == NetworkType.RGG or netType == NetworkType.RGG_FIXED_DEGREE):
                    pos_layout = nx.get_node_attributes(graph, "pos")
                else:
                    pos_layout = nx.circular_layout(graph)
                nx.draw(graph, pos_layout, node_color=decisionColors, with_labels=True)
                plt.show()

            if not 2 in nx.get_node_attributes(graph, 'opinion').values():
                if DEBUG: print("Consensus for option 1 reached at timestep " + str(t) )
                convergence_times[rep] = t
                break
            
        
        ## Plotting the graph with colored nodes
        if DEBUG:
            decisionColors=[]
            for n in graph.nodes():
                if (graph.nodes[n]['opinion']==1): decisionColors.append('r') 
                elif  (graph.nodes[n]['opinion']==2): decisionColors.append('g')
                else: decisionColors.append('y')
            if (netType == NetworkType.RGG or netType == NetworkType.RGG_FIXED_DEGREE):
                pos_layout = nx.get_node_attributes(graph, "pos")
            else:
                pos_layout = nx.circular_layout(graph)
            nx.draw(graph, pos_layout, node_color=decisionColors, with_labels=True)
            plt.show()
    
    os.makedirs(os.path.dirname(outputTxtFile), exist_ok=True)
    with open(outputTxtFile, 'w') as outstream:
        outstream.write('\n'.join(str(item) for item in convergence_times)) # write each item on a new line
        if DEBUG: print("Written file " + str(outputTxtFile))
    
    if DEBUG: 
        print("Process Terminated")
