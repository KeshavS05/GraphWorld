from collections import defaultdict
import sys
import time
from itertools import product
class JSGraph():
   def __init__(self, V, weight_after):
      self.weight_after = weight_after
      '''
      Instantiate JSG Graph with the number of vertices of the graph.
      '''
      # EG variables
      self.V = V # vertices/nodes
      self.graph_1d = [[0 for column in range(V)] for row in
                     range(V)] # used adjM for Environment Graph (EG)
         
      # JSG Variables
      self.nodesets = [] # nodesets for JSG
      #self.graph = [[[] for column in range(V)] for row in
                     #range(V)] # used adjMatrix for Joint State Graph (JSG)
      self.graph = {}
      
   # get adjMatrix of JSG
   def get_adjMatrix_2D_JSG(self):
      return self.graph

   # add adjMatrix of joint state graph
   def addEdge_2d(self, current_positions, new_positions, w):#u,v,x,y, w):
        '''
        The Graph will be bidirectional and assumed to have positive weights.
        '''
      # # this works for adjList
      # if [w,(x,y)] not in self.graph_2d[(u,v)]:
      #     self.graph_2d[(u,v)].append([w,(x,y)])
      # if [w,(u,v)] not in self.graph_2d[(x,y)]:
      #     self.graph_2d[(x,y)].append([w,(u,v)])  
      
      # this works for adjMatrix 
      #if [w,(x,y)] not in self.graph[u][v]:
         #self.graph[u][v].append([w,(x,y)])
      #if [w,(u,v)] not in self.graph[x][y]:
         #self.graph[x][y].append([w,(u,v)])
        
        if current_positions in self.graph and [w, new_positions] not in self.graph[current_positions]:#[u][v]:
            self.graph[current_positions].append([w, new_positions])
        elif current_positions not in self.graph:
            self.graph[current_positions] = [[w, new_positions]]
        
        if new_positions in self.graph and [w, current_positions] not in self.graph[new_positions]:#[u][v]:
            self.graph[new_positions].append([w, current_positions])
        elif new_positions not in self.graph:
            self.graph[new_positions] = [[w, current_positions]]

   
   # add modes of JSG
   def add_nodesets(self, n_agents):
      
      # for u in range(self.V):
      #    for v in range(self.V):
      #          if (u,v) not in self.nodesets:
      #             self.nodesets.append((u,v))
      for s in product([i for i in range(self.V)], repeat=n_agents):
          if s not in self.nodesets:
            self.nodesets.append(s)
      return self.nodesets
   
   
   
   # calcualte edge cost
   def edge_cost(self,set1, set2, support=False):
      dist = 0
      for i in range(len(set1)):
         if type(self.graph_1d[set1[i]][set2[i]])==list:
            dist_iw,_= self.graph_1d[set1[i]][set2[i]]
         else:
            dist_iw = self.graph_1d[set1[i]][set2[i]]  
               
         if dist_iw == float('inf'):
            return float('inf')   
         dist += dist_iw
      if (support):
         return dist/2  # if has support edge cost reduced by half   
      return dist    
   

   # convert Environment Graph to Joint State Space Graph    
   def construct_JSG(self, n_agents):
      for set1 in self.nodesets:
         for set2 in self.nodesets:
               moving_labels, staying_labels = [], []
               for ind, (i, j) in enumerate(zip(set1, set2)):
                  if i == j:
                     staying_labels.append(ind)
                  else:
                     moving_labels.append(ind)
               #print((i,j),(w,k))
               #A constant B moving (u,v)-(i,j) and (x,y)-(w,k)
               invalid = False
               for a in moving_labels:
                  if self.graph_1d[set1[a]][set2[a]]==0 or self.graph_1d[set1[a]][set2[a]]==float('inf'):
                     invalid = True
               if not invalid:
                  for a in moving_labels:
                     support_node_jk = []
                     if type(self.graph_1d[set1[a]][set2[a]])==list:
                           _,support_node_jk = self.graph_1d[set1[a]][set2[a]]
                     support = False
                     for a in staying_labels:
                           if set1[a] in support_node_jk:
                              support = True
                     dist = self.edge_cost(set1,set2,support)
                     if dist!=0 and dist!=float('inf'): 
                           self.addEdge_2d(set1,set2,dist)  
                     
                     
   # transofrm Environment Graph to Joint State Space Graph                    
   def trans_Env_To_JSG(self, n_agents):
      self.add_nodesets(n_agents)
      self.construct_JSG(n_agents)

# from collections import defaultdict
# import sys

# class JSGraph():
#     def __init__(self, V):
#         '''
#         Instantiate JSG Graph with the number of vertices of the graph.
#         '''
#         # EG variables
#         self.V = V # vertices/nodes
#         self.source = (None, None) # start position
#         self.destination = (None, None) # goal position
#         self.graph_1d = [[0 for column in range(V)] for row in
#                       range(V)] # used adjM for Environment Graph (EG)
          
#         # JSG Variables
#         self.nodesets = [] # nodesets for JSG
#         self.graph = [[[] for column in range(V)] for row in
#                       range(V)] # used adjMatrix for Joint State Graph (JSG)
        
#     # get adjMatrix of JSG
#     def get_adjMatrix_2D_JSG(self):
#         return self.graph
  
#     # add adjMatrix of joint state graph
#     def addEdge_2d(self, u, v,x,y, w):
#         '''
#         The Graph will be bidirectional and assumed to have positive weights.
#         '''
#         # # this works for adjList
#         # if [w,(x,y)] not in self.graph_2d[(u,v)]:
#         #     self.graph_2d[(u,v)].append([w,(x,y)])
#         # if [w,(u,v)] not in self.graph_2d[(x,y)]:
#         #     self.graph_2d[(x,y)].append([w,(u,v)])  
        
#         # this works for adjMatrix 
#         if [w,(x,y)] not in self.graph[u][v]:
#             self.graph[u][v].append([w,(x,y)])
#         if [w,(u,v)] not in self.graph[x][y]:
#             self.graph[x][y].append([w,(u,v)])
    
#     # add modes of JSG
#     def add_nodesets(self):
        
#         for u in range(self.V):
#             for v in range(self.V):
#                 if (u,v) not in self.nodesets:
#                     self.nodesets.append((u,v))
#         return self.nodesets
#     # calcualte edge cost
#     def edge_cost(self,i,j,w,k,support=False):
#         dist = 0
        
#         if type(self.graph_1d[i][w])==list:
#             dist_iw,_= self.graph_1d[i][w]
#         else:
#             dist_iw = self.graph_1d[i][w]     
                  
#         if type(self.graph_1d[j][k])==list:
#             dist_jk,_ = self.graph_1d[j][k]
#         else:
#             dist_jk = self.graph_1d[j][k] 
            
#         dist = dist_iw + dist_jk
#         if dist_iw == float('inf') or dist_jk==float('inf'):
#             return float('inf')
#         if (support):
#             return  int(dist/2)    # if has support edge cost reduced by half   
#         return dist  
      
 
#     # convert Environment Graph to Joint State Space Graph    
#     def construct_JSG(self):
#         for (i,j) in self.nodesets:
#             for (w,k) in self.nodesets:
#                 #print((i,j),(w,k))
#                 #A constant B moving (u,v)-(i,j) and (x,y)-(w,k)
#                 if i==w and j!=k and self.graph_1d[j][k]!=0 and self.graph_1d[j][k]!=float('inf') : # k belongs to N of j
#                     support_node_jk = []
#                     if type(self.graph_1d[j][k])==list:
#                         _,support_node_jk = self.graph_1d[j][k]
                        
#                     #print("??????????????????????????????????",support_node_jk )   
#                     if i in support_node_jk:
#                         dist = self.edge_cost(i,j,w,k,support=True)
#                     else:
#                         dist = self.edge_cost(i,j,w,k)
#                     ############################
#                     if dist!=0 and dist!=float('inf'): 
#                         print("A Constant B moving")
#                         print((i,w),(j,k),dist)  
#                         self.addEdge_2d(i,j,w,k,dist)
#                 # B constant, A moving
#                 elif i!=w and j==k and self.graph_1d[w][i]!=0 and self.graph_1d[w][i]!=float('inf') :
#                     support_node_iw = []
#                     if type(self.graph_1d[i][w])==list:
#                         _,support_node_iw= self.graph_1d[i][w]  
                           
#                     if j in support_node_iw:
#                         dist = self.edge_cost(i,j,w,k,support=True)
#                     else:
#                         dist = self.edge_cost(i,j,w,k)
#                     ############################### 
#                     if dist!=0 and dist!=float('inf'):  
#                         print("B Constant A moving")
#                         print((i,w),(j,k),dist)   
#                         self.addEdge_2d(i,j,w,k,dist)  
#                 elif self.graph_1d[j][k]!=0 and self.graph_1d[j][k]!=float('inf') and self.graph_1d[i][w]!=0 and self.graph_1d[i][w]!=float('inf') :   
#                     dist= self.edge_cost(i,j,w,k)
#                     ############################
#                     if dist!=0 and dist!=float('inf'): 
#                         print("Both moving: from privious node to next node")
#                         print((i,w),(j,k),dist) 
#                         self.addEdge_2d(i,j,w,k,dist)  
                        
                        
#     # transofrm Environment Graph to Joint State Space Graph                    
#     def trans_Env_To_JSG(self):
#         self.add_nodesets()
#         self.construct_JSG()         