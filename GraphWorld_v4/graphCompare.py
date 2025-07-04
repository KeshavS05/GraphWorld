from randomGraphGenerator import RandomGraph
from our_jsg import JSGraph
import time
import argparse
from itertools import product

# print and return the shortest path from source to goal
def printSolution(V, dist_2d, S11,Sgg, n_agents):
    total_shortest_path_cost = 0
    print("Source\tAgent_Position\tDistance_from_Source")
    b = False
    for key in dist_2d:
        if key == Sgg:   
            total_shortest_path_cost = int(dist_2d[key])
            b = True
    if not b:
        total_shortest_path_cost = float('inf')
    return total_shortest_path_cost

# minDistance_2d() returns the vertex with minimum distance value,
# from the set of vertices not yet included in shortest path tree

def minDistance_2d(V, dist_2d, graph, seen_2d):
    min_d = float('inf')
    for key in graph:
        if key not in seen_2d and key in dist_2d and dist_2d[key] < min_d:
            min_d = dist_2d[key]
            min_vertex = key
    return min_vertex

# shortest_path_cost() uses custo dijkstra to calculate the shortest path cost from source to goal
def shortest_path_cost(V, graph, iter, S11, Sgg):     
    #dist_2d = [[float('inf') for column in range(V)] for row in range(V)]
    dist_2d = {S11:0}      
    seen_2d = set()
    path = []
    #dist_2d[S11[0]][S11[1]] = 0   
    n_agents = 0
    for _ in range(iter):
        
        min_vertex = minDistance_2d(V, dist_2d, graph, seen_2d)
        if n_agents == 0:
            n_agents = len(min_vertex)
        seen_2d.add(min_vertex)

        for weight, nxt_node in graph[min_vertex]:
            if nxt_node not in seen_2d and nxt_node in dist_2d and min_vertex in dist_2d and dist_2d[nxt_node] > dist_2d[min_vertex] + weight:
                dist_2d[nxt_node] = dist_2d[min_vertex] + weight
            elif nxt_node not in seen_2d and nxt_node not in dist_2d and min_vertex in dist_2d:
                dist_2d[nxt_node] = dist_2d[min_vertex] + weight 
            
    return printSolution(V, dist_2d, S11,Sgg, n_agents)

# compare_JSG_and_CJSG() compares the time taken by JSG and CJSG for same EG
def compareGraphs_JSG_and_CJSG(V,E, S11, Sgg, adj_ws, adj_ns, n_agents):
    ############ Joint State Graph (JSG) #################
    jsg = JSGraph(V)
    jsg.source = S11
    jsg.destination = Sgg
    jsg.graph_1d = adj_ws
    #print("adj_ws: ", adj_ws)
    jsg_gc_start = time.time()
    jsg.trans_Env_To_JSG(n_agents)
    #print(jsg.nodesets)
    jsg_gc_end = time.time()
    
    jsg_spc_start = time.time()
    jsg_spc= shortest_path_cost(V, graph=jsg.get_adjMatrix_2D_JSG(), iter=V*V, S11=S11, Sgg=Sgg)
    jsg_spc_end = time.time()
  

  

    ########### Time Comparision JSG and CJSG ##########
    print("----------------JSG Outputs--------------")
    print("Graph Construction Time: {}".format(jsg_gc_end-jsg_gc_start))
    print("Shortest Path Cost: {}".format(jsg_spc))
    print("Shortest Path Time (ms): {}".format((jsg_spc_end - jsg_spc_start)))
    print("Total Time Taken: {}".format((jsg_spc_end - jsg_spc_start)+(jsg_gc_end-jsg_gc_start)))
    print("----------------------------------------")
    
# randomGraphCompare() generates random Environment Graph (EG) 
# and compares the time taken by JSG and CJSG for same EG
def randomGraphCompare(input_vertices, input_riskedges):
    
    # Random Graph with risky edges and support nodes
    V = input_vertices
    rG = RandomGraph(V)         
    rG.generateRandomGraph()        
    E = rG.countEdgesEG()
    # risky edges: 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
    risky_edges = int(0.2*E)
    rG.generateRGWithSupportNodesAndRiskyEdges(risky_edges)
    adj_ws = rG.getRGWithRiskEdgesAndSupportNodes() # EG with risky edges and support nodes
    adj_ns = rG.getRGWithNoRiskEdgesAndSupportNodes() # EG with no risky edges and support nodes
    
    print("--------------EG Inputs-----------------")
    print("Nodes: {}".format(V))
    print("Edges: {}".format(E))
    print("Risk Edges: {}".format(rG.getTotalRiskyEdges()))
    print("----------------------------------------")

    S, G = 0,(V-1) # start and goal node in EG
    S11 = (S,S)    # start position in JSG and CJSG
    Sgg = (G,G)    # goal position in JSG and CJSG
    
    # comapre jsg vs cjsg
    compareGraphs_JSG_and_CJSG(V, E, S11, Sgg, adj_ws, adj_ns, )
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JSG and CJSG Comparision for Random Graphs')
    parser.add_argument('-n', '--nodes', type=int, required=True, help='Path to the input file')
    parser.add_argument('-r', '--riskedge', type=float, required=True, help='Path to the output file')
    parser.add_argument('-a', '--n_agents', type=float, required=True, help='Path to the output file')
    args = parser.parse_args()
    input_vertices = args.nodes
    input_riskedges_ratio = args.riskedge
    n_agents = args.n_agents
    randomGraphCompare(input_vertices, input_riskedges_ratio, n_agents)

# from randomGraphGenerator import RandomGraph
# from our_jsg import JSGraph
# import time
# import argparse

# # print and return the shortest path from source to goal
# def printSolution(V, dist_2d, S11,Sgg):
#     total_shortest_path_cost = 0
#     print("Source\tAgent_Position\tDistance_from_Source")
#     for u in range(V):   
#         for v in range(V):
#             if (u,v) == Sgg:
#                 print((S11[0],S11[1]),(u,v), dist_2d[u][v]) 
#                 total_shortest_path_cost = int(dist_2d[u][v])
#     return total_shortest_path_cost

# # minDistance_2d() returns the vertex with minimum distance value,
# # from the set of vertices not yet included in shortest path tree

# def minDistance_2d(V, dist_2d, seen_2d):
#     min_d = float('inf')
#     for u in range(V):
#         count = 0
#         for v in range(V):
#             count = count+1
#             #print((u,v))
#             if (u,v) not in seen_2d and dist_2d[u][v] < min_d:
#                 min_d = dist_2d[u][v]
#                 min_vertex =u,v
#     return min_vertex

# # shortest_path_cost() uses custo dijkstra to calculate the shortest path cost from source to goal
# def shortest_path_cost(V, adjList_2D, iter, S11, Sgg):     
#     dist_2d = [[float('inf') for column in range(V)] for row in
#                         range(V)]      
#     seen_2d = set()
#     dist_2d[S11[0]][S11[1]] = 0    
#     for _ in range(iter):
#         print("\n")
#         print("Iteration",_)
#         u,v = minDistance_2d(V, dist_2d, seen_2d)
#         seen_2d.add((u,v))
#         #print("seen_node, next_node, edge_cost, dist_from_source")
#         #print(self.H[(u,v)])
#         count = 0
#         print((u,v))
#         #print((u,v), adjList_2D[u][v])
#         for weight, nxt_node in adjList_2D[u][v]:
#             #print(int(weight), nxt_node)
           
#             print("Inside Iteration",count)
#             count= count+1
#             #print((u,v), nxt_node, weight)
#             x,y = nxt_node 
#             if nxt_node not in seen_2d and dist_2d[x][y] > dist_2d[u][v] + weight:
#                 #print(">>>>>>>>>>>>>>>>>>>>>>>>")
#                 dist_2d[x][y] = dist_2d[u][v] + weight
#     return printSolution(V, dist_2d, S11,Sgg)

# # compare_JSG_and_CJSG() compares the time taken by JSG and CJSG for same EG
# def compareGraphs_JSG_and_CJSG(V,E, S11, Sgg, adj_ws, adj_ns):
#     ############ Joint State Graph (JSG) #################
#     jsg = JSGraph(V)
#     jsg.source = S11
#     jsg.destination = Sgg
#     jsg.graph_1d = adj_ws
    
#     jsg_gc_start = time.time()
#     jsg.trans_Env_To_JSG() 
#     jsg_gc_end = time.time()
    
#     jsg_spc_start = time.time()
#     jsg_spc= shortest_path_cost(V, adjList_2D=jsg.get_adjMatrix_2D_JSG(), iter=V*V, S11=S11, Sgg=Sgg)
#     jsg_spc_end = time.time()
  
  

#     ########### Time Comparision JSG and CJSG ##########
#     print("----------------JSG Outputs--------------")
#     print("Graph Construction Time: {}".format(jsg_gc_end-jsg_gc_start))
#     print("Shortest Path Cost: {}".format(jsg_spc))
#     print("Shortest Path Time (ms): {}".format((jsg_spc_end - jsg_spc_start)))
#     print("Total Time Taken: {}".format((jsg_spc_end - jsg_spc_start)+(jsg_gc_end-jsg_gc_start)))
#     print("----------------------------------------")
    
# # randomGraphCompare() generates random Environment Graph (EG) 
# # and compares the time taken by JSG and CJSG for same EG
# def randomGraphCompare(input_vertices, input_riskedges):
    
#     # Random Graph with risky edges and support nodes
#     V = input_vertices
#     rG = RandomGraph(V)         
#     rG.generateRandomGraph()        
#     E = rG.countEdgesEG()
#     # risky edges: 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
#     risky_edges = int(0.2*E)
#     rG.generateRGWithSupportNodesAndRiskyEdges(risky_edges)
#     adj_ws = rG.getRGWithRiskEdgesAndSupportNodes() # EG with risky edges and support nodes
#     adj_ns = rG.getRGWithNoRiskEdgesAndSupportNodes() # EG with no risky edges and support nodes
    
#     print("--------------EG Inputs-----------------")
#     print("Nodes: {}".format(V))
#     print("Edges: {}".format(E))
#     print("Risk Edges: {}".format(rG.getTotalRiskyEdges()))
#     print("----------------------------------------")

#     S, G = 0,(V-1) # start and goal node in EG
#     S11 = (S,S)    # start position in JSG and CJSG
#     Sgg = (G,G)    # goal position in JSG and CJSG
    
#     # comapre jsg vs cjsg
#     compareGraphs_JSG_and_CJSG(V, E, S11, Sgg, adj_ws, adj_ns)
    
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='JSG and CJSG Comparision for Random Graphs')
#     parser.add_argument('-n', '--nodes', type=int, required=True, help='Path to the input file')
#     parser.add_argument('-r', '--riskedge', type=float, required=True, help='Path to the output file')
#     args = parser.parse_args()
#     input_vertices = args.nodes
#     input_riskedges_ratio = args.riskedge
#     randomGraphCompare(input_vertices, input_riskedges_ratio)


