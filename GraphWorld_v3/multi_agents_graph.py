
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple
import time
from PIL import Image
import io
import imageio
import random
import scipy as sp

def generate_random_connected_graph(n_nodes):
    
    assert n_nodes > 1, "Number of nodes must be greater than 1"

    # Generate nodes
    nodes = {i: (i) for i in range(n_nodes)}

    # Generate edges
    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    random.shuffle(all_edges)

    # Start with a tree structure to ensure all nodes are connected
    tree_edges = all_edges[:n_nodes - 1]

    # Randomly decide on remaining edges
    remaining_edges = np.random.randint(0, n_nodes * (n_nodes - 1) / 2 - len(tree_edges))
    additional_edges = all_edges[n_nodes - 1 : n_nodes - 1 + remaining_edges]

    # Combine tree edges and additional edges
    edges = tree_edges + additional_edges

    # Convert edges to dictionary
    edges = {edge: 1 for edge in edges}

    return nodes, edges



def generate_random_connected_graph(n_nodes, n_edges):
    start_node = 0
    goal_node = n_nodes - 1
    assert n_nodes > 1, "Number of nodes must be greater than 1"
    assert n_edges <= (n_nodes * (n_nodes - 1) / 2), "Number of edges cannot exceed the maximum possible"
    assert 0 <= start_node < n_nodes, "Start node is out of range"
    assert 0 <= goal_node < n_nodes, "Goal node is out of range"
    assert start_node != goal_node, "Start node and goal node cannot be the same"


    # # Generate nodes
    nodes = {i: i for i in range(n_nodes)}

    # Generate all possible edges
    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    # Randomly select a subset of edges
    random.shuffle(all_edges)
    selected_edges = all_edges[:n_edges]

    # Convert edges to dictionary
    edges = {edge: 1 for edge in selected_edges}

    return nodes, edges
    
    ## Generate nodes
    nodes = {i: i for i in range(n_nodes)}

    # Generate all possible edges
    all_edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes) if i != j]

    # Select one edge for the start node and one for the goal node
    edges_start = [(start_node, j) for j in range(n_nodes) if j != start_node]
    edges_goal = [(i, goal_node) for i in range(n_nodes) if i != goal_node]

    start_edge = random.choice(edges_start)
    goal_edge = random.choice(edges_goal)

    # Ensure the selected edges for start and goal nodes are not the same
    while start_edge == goal_edge:
        start_edge = random.choice(edges_start)
        goal_edge = random.choice(edges_goal)

    # Remove the selected edges for start and goal nodes from all_edges
    all_edges.remove(start_edge)
    all_edges.remove(goal_edge)

    # Randomly select remaining edges
    random.shuffle(all_edges)
    selected_edges = all_edges[:n_edges-2]  # -2 because we already have 2 edges

    # Convert edges to dictionary and add the selected edges for start and goal nodes
    edges = {edge: 1 for edge in [start_edge, goal_edge] + selected_edges}

    return nodes, edges
    
    



def generate_fully_connected_graph(n_nodes):
    assert n_nodes > 1, "Number of nodes must be greater than 1"

    # Generate nodes
    nodes = {i: (i) for i in range(n_nodes)}
    
    # Generate edges: every node is connected to every other node
    edges = {(i, j): 1 for i in range(n_nodes) for j in range(i+1, n_nodes)}
    
    return nodes, edges


class Graph:
    def __init__(self):
        # self.graph = nx.Graph()
        # self.graph.add_nodes_from(nodes)
        # self.graph.add_edges_from(edges)
        self.graph = None
        self.edge_indices = {}
         
        # self.nodes_list = sorted(self.graph.nodes) # from adj (contains nodes only)
        # self.edges_list = sorted(self.graph.edges)  # from adj (contains edges only)
        self.risk_edges = [] # exitsting risk edges   
        self.safe_edges = [] # existing safe edges
        self.risk_edges_with_support_nodes_dict = {}
        
        
        self.graph_adjacency_matrix = None
        self.graph_adjM_original = None
        self.graph_adjM_with_edge_cost = None
        self.graph_adjM_with_reduced_cost = None
        
    def number_of_nodes(self):
        num_nodes = list(self.graph.nodes)
        return len(num_nodes)
    
    def number_of_edges(self):
        num_edges = list(self.graph.edges)
        return len(num_edges)
        
    def neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    # plain graph adjacency matrix
    def adjacency_matrix(self):
        adj_matrix_array = nx.adjacency_matrix(self.graph)
        self.graph_adjacency_matrix = adj_matrix_array
        return self.graph_adjacency_matrix
    
    # every node to every node matrix with just edge cost (no support nodes)
    def env_graph_adjM_with_edge_cost(self):
        adj_matrix_list = nx.adjacency_matrix(self.graph).todense().tolist()
        adj_matrix_list =  [[10 if adj_matrix_list[i][j]!=0 else element for j, element in enumerate(row)] for i, row in enumerate(adj_matrix_list)]
        self.graph_adjM_with_edge_cost = adj_matrix_list
        return self.graph_adjM_with_edge_cost
    
    # every node to node matrix withe edge cost and support nodes
    def env_graph_adjM_original(self):
        if self.graph_adjM_with_edge_cost is None:
            # Initialize the adjacency matrix
            self.adjacency_matrix() # ndarray
            # Initialize the adjacency matrix with edge cost
            self.env_graph_adjM_with_edge_cost() # list
            
            
            # Determine the number of edges to modify
            num_edges_to_modify = int(self.number_of_edges()/2)
            adj_matrix_array = self.graph_adjacency_matrix
            # calculate randome edges to modify with another random value
            node_values = list(range(adj_matrix_array.shape[0]))
            
            # Randomly select the edges to modify
            edges_to_modify = set() # edges to modify
            edges_to_modify_dict = {} # modified edges with supporte node values
            
        # Randomly assign the random list of two values to the selected edges
            while len(edges_to_modify) < num_edges_to_modify:
            #for _ in range(num_edges_to_modify):
                non_zero_indices = np.nonzero(adj_matrix_array)
                valid_edges = list(zip(non_zero_indices[0], non_zero_indices[1]))
                random_edge = random.choice(valid_edges)
                random_values = np.random.choice(node_values, size=2, replace=False).tolist()
                if random_edge not in edges_to_modify and random_edge[::-1] not in edges_to_modify:
                    edges_to_modify.add(random_edge)
                    if random_edge in edges_to_modify:
                        edges_to_modify_dict[random_edge] = random_values
                        
            self.risk_edges = list(edges_to_modify_dict)
            self.risk_edges_with_support_nodes_dict = edges_to_modify_dict
        
        # Randomly assign the random list of two values to the selected edges
        # adj_matrix_list = nx.adjacency_matrix(self.graph).todense().tolist()
            
        # Add with a fixed edge cost of 10 for all edges
            adj_matrix_list = self.graph_adjM_with_edge_cost 
            print("edges to modify:",sorted(edges_to_modify))  
            print("edges to support nodes dict",sorted(edges_to_modify_dict.items()))  
            
            visited_edges = set()
            for i in range(len(node_values)):
                for j in range(len(node_values)):
                    
                    if i!=j and (i, j) not in visited_edges:
                        # Access element at position [i][j]
                        # print(f"Element at [{i}][{j}] is {adj_matrix_list[i][j]}")
                        if (i, j) in edges_to_modify_dict:
                            
                            # print(f"Elemeent at edge ({i}, {j}) are random values {edges_to_modify_dict[(i, j)]}")
                            adj_matrix_list[i][j] = [10, edges_to_modify_dict[(i, j)]]  
                            adj_matrix_list[j][i] = [10, edges_to_modify_dict[(i, j)]] 
                            visited_edges.add((i, j))   
                            visited_edges.add((j, i))   
                                    
                        # print(f"After update, element at [{i}][{j}] is {adj_matrix_list[i][j]}")
            # print("heheheheheheh",adj_matrix_list)
            self.graph_adjM_original = adj_matrix_list
            
        return self.graph_adjM_original
    # using adjMatrix calculate node to node combinations as edges
    def non_existenting_and_existing_edges(self):
        adj_edges_list = set() # from adj(existing and non-existing edges)
        adj_matrix_list = self.env_graph_adjM_original()
        for i in range(len(adj_matrix_list)): # row
            for j in range(len(adj_matrix_list[i])): # column
                adj_edges_list.add((i,j))
        return sorted(adj_edges_list) # VVI: sorted is important to get the same order of edges everytime
    

    # every node to every edge matrix with  reduced edge cost for the risk edges with support nodes
    def env_graph_adjM_with_reduced_cost(self):
        if self.graph_adjM_with_reduced_cost is None:
            # Create a new dictionary for the node-edge costs
            node_edge_costs = {} # node to edge costs
            adj_matrix_list = self.env_graph_adjM_original()
            adj_edges_list = set() # from adj(existing and non-existing edges)
            
            # Iterate over the adjacency matrix
            #print("---------------------------  Adjacency Matrix ---------------------------")
            
            # print(adj_matrix_list)
            # print(self.risk_edges_with_support_nodes_dict.items())
            for i in range(len(adj_matrix_list)): # row
                for j in range(len(adj_matrix_list[i])): # column
                    adj_edges_list.add((i,j))
                    # add all edges (existing+ non-existing) to a set
                # print("----------------------")
                    # If the element is a list, it means the edge has support nodes
                    if isinstance(adj_matrix_list[i][j], list):
                        
                        # print("Yesy yesy yesy yesy")
                        # The base cost of the edge is the first element of the list
                        base_cost = adj_matrix_list[i][j][0]
                        # The list of support nodes is the second element of the list
                        support_nodes = adj_matrix_list[i][j][1]
                        # print(self.risk_edges_with_support_nodes_dict.items())
                        # print((i,j), support_nodes)
                        for edges, support_nodes in self.risk_edges_with_support_nodes_dict.items():
                            if (i,j) == edges:
                                for support_node in support_nodes:
                                    if (support_node, (i, j) not in node_edge_costs):
                                        # print(f"support node {support_node} for edge ({i}, {j})")
                                        # print(f"support node {support_node} for edge ({j}, {i})")
                                        node_edge_costs[(support_node, (i, j))] = 5
                                        node_edge_costs[(support_node, (j, i))] = 5

                        # # Calculate the cost reduction for each support node
                        # for node in support_nodes:
                        #     # Let's assume a support node reduces the cost by 2
                        #     cost_reduction = 5
                        #     # The new cost of the edge is the base cost minus the cost reduction
                        #     new_cost = base_cost - cost_reduction
                        #     # Update the node-edge cost in the dictionary
                        #     node_edge_costs[(node, (i, j))] = new_cost
                    else:
                        # print("no no no no no")
                        # If the element is not a list, just use the cost from the adjacency matrix
                        node_edge_costs[(i, (i, j))] = adj_matrix_list[i][j]
                        
            # sort the edges list
            adj_edges_list = sorted(adj_edges_list)
            # Map edge tuples to their indices in edge_list 
            edge_indices = {edge: index for index, edge in enumerate(adj_edges_list)}
            # Initialize the matrix
            matrix = np.full((self.number_of_nodes(),len(adj_edges_list)), np.inf)
            
            # # Update the matrix with costs from the dictionary
            #print(adj_edges_list)
            for i in range(len(adj_matrix_list)): # row
                for j in range(len(adj_edges_list)):
                    #print(i,adj_edges_list[j])  
                    for node_edge, cost in node_edge_costs.items():
                            node, edge = node_edge
                            if node==i and edge == adj_edges_list[j]:
                                matrix[node, edge_indices[edge]] = cost
                        
            # print("Node Edeg Cost dict:",sorted(node_edge_costs.items()))
            # print("Risk Edges with Support nodes dict",self.risk_edges_with_support_nodes_dict.items())            
            # print("adjM_reduced_cost",matrix)
            self.graph_adjM_with_reduced_cost = matrix
    
        return self.graph_adjM_with_reduced_cost
                
    def fixed_start_goal(self):
        nodes = list(self.graph.nodes)
        return nodes[0]
    
    def fixed_end_goal(self):
        nodes = list(self.graph.nodes)
        return 4
       # return nodes[-1]

    def random_start_goal(self):
        nodes = list(self.graph.nodes)
        return np.random.choice(nodes, 2, replace=False)

    def shortest_path(self, start, goal):
        return nx.shortest_path(self.graph, start, goal)

    def draw(self, agent_positions):
        color_map = []
        for node in self.graph:
            if node in agent_positions:
                color_map.append('blue')
            else: 
                color_map.append('red')
        nx.draw(self.graph, with_labels=True, node_color=color_map)
        #nx.spring_layout(self.graph.graph)
        plt.show()


class MultiAgentGraphWorldEnv(gym.Env):
    def __init__(self, graph, n_agents):
        super(MultiAgentGraphWorldEnv, self).__init__()
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.n_agents = n_agents
        self.agent_positions = [0]*self.n_agents
        self.agent_rewards = 0
        self.centralized = True #default is True
        
        self.frames = []  # for storing frames
        self.node_positions = nx.spring_layout(self.graph.graph)
        
        self.env_graph_adjM_original = None
        self.env_graph_adjM_with_edge_cost = None
        self.env_graph_adjM_with_reduced_cost = None
        self.edge_index_map = {}
        

        # Define action and observation space  
        # They must be gym.spaces objects 
        self.action_space = spaces.Tuple([spaces.Discrete(self.n_nodes) for _ in range(self.n_agents)])
        self.observation_space = spaces.Tuple([spaces.Discrete(self.n_nodes) for _ in range(self.n_agents)])
        
    def get_observation_space(self):  
        #one_hot_agent_positions = self.one_hot_encode_agent_positions()
        # adjM_with_edge_cost = self.graph.env_graph_adjM_with_edge_cost()
        # adjM_with_reduced_edge_cost = self.graph.env_graph_adjM_with_reduced_cost()
        #print("Inside get_observation_space")
        adjM_with_edge_cost = self.env_graph_adjM_with_edge_cost
        adjM_with_reduced_edge_cost = self.env_graph_adjM_with_reduced_cost
        # print(adjM_with_edge_cost)
        # print(adjM_with_reduced_edge_cost)
        
        
        
        full_obs = []
        # preporcess agent positions
        agent_position_one_hot_total = self.one_hot_encode_agent_positions()
        # preprocess adjM_with_edge_cost
        # Iterate over each node and its neighbors
        for node in range(np.array(adjM_with_edge_cost).shape[0]):
            for neighbor in range(np.array(adjM_with_edge_cost).shape[0]):
                if node == neighbor:
                    adjM_with_edge_cost[node][neighbor] = 0
                elif node!=neighbor and adjM_with_edge_cost[node][neighbor] == 0:
                    adjM_with_edge_cost[node][neighbor] = 100
                    
        #print("Preprocessed adjM_with_edge_cost",adjM_with_edge_cost)
        # preprocess adjM_with_reduced_edge_cost
        # this will be done at end (do it here later)
        
        #agent_position_one_hot_total= agent_position_one_hot_total.reshape(-1, 1)
        #print(f"one hot: {agent_position_one_hot_total}") 
        #print(f"one hot shape {agent_position_one_hot_total.shape} adjM1 shape {np.array(adjM_with_edge_cost).shape} adjM2 shape {adjM_with_reduced_edge_cost.shape}")
        agent_obs_concat = np.concatenate((agent_position_one_hot_total.reshape(-1, 1),np.array(adjM_with_edge_cost).reshape(-1, 1), adjM_with_reduced_edge_cost.reshape(-1, 1)), axis=0)
        agent_obs_concat.flatten().tolist() 
        #print(f"aoc shape: {agent_obs_concat.shape}")
        full_obs.append(agent_obs_concat)
            
            
            
        update_full_obs = np.array(full_obs).flatten().tolist()  
        #print("update_full_obs", np.array(full_obs).flatten().shape)
        update_full_obs = [100 if np.isinf(x) else x for x in update_full_obs]      
        return  update_full_obs
    
    def get_action_space(self): 
        agent_positions = self.agent_positions
        action_space_on_hot = np.zeros((self.n_agents, self.n_nodes+1), dtype=int)
        for i in range(len(action_space_on_hot)):
            action_space_on_hot[i, agent_positions[i]] = 1 
        return action_space_on_hot
             
    def one_hot_encode_agent_positions(self):
        agent_positions_one_hot = np.zeros((self.n_agents, self.n_nodes))
        for agent_idx, position in enumerate(self.agent_positions):
            agent_positions_one_hot[agent_idx, position] = 1
        return agent_positions_one_hot
    
    def sample_action_space(self):
        sample_actions = np.random.choice(self.n_nodes, size=2, replace=False)
        print("+++++++++++++++=========================================")
        print("sample_actions", sample_actions)
        agent_positions_one_hot_after_sample_action = np.zeros((self.n_agents, self.n_nodes))
        for agent_idx, position in enumerate(sample_actions):
            agent_positions_one_hot_after_sample_action[agent_idx, position] = 1
        print("sample_actions on hot", agent_positions_one_hot_after_sample_action)
        print("+++++++++++++++=========================================")
        return agent_positions_one_hot_after_sample_action
     
     
    
    def take_action(self, actions):
        agent_positions_one_hot_after_action = np.zeros((self.n_agents, self.n_nodes))
        for agent_idx, position in enumerate(actions):
            agent_positions_one_hot_after_action[agent_idx, position] = 1
        return agent_positions_one_hot_after_action
        
        
    def step(self, actions):
        # this information is full observation information (this is current state)
        reward = 0
        ############ recostruction of 1. agents actions 2. agents current position  3. node to node matrix with orginal cost 4. node to edge matrix with reduced cost for easy mainupulation
        ## A. reconstruction agents actions

        actions= np.argmax(actions, axis=1)
   
        
        ## B. reconstruction from current observation
        current_state = self.get_observation_space()
        current_state = np.array(current_state).reshape(1, -1)
        #print("current_state", current_state[0])
       #print(current_state[0])
        current_state = current_state[0]
        #1. agent positions
        agent_positions_sub_array = current_state[0 : 0 + np.prod((2,5))]
        agent_positions = agent_positions_sub_array.reshape((2,5))
   
        # 2. adjM with edge cost reconstruction from the 1D array
        adjM_with_edge_cost_sub_array = current_state[10 : 10 + np.prod((5,5))]
        adjM_with_edge_cost = adjM_with_edge_cost_sub_array.reshape((5,5))
        
        # 3.adjM with reduced edge cost from the reconstruction of 1D array
        adjM_with_reduced_edge_cost_sub_array = current_state[35: 35 + np.prod((5,25))]
        adjM_with_reduced_edge_cost = adjM_with_reduced_edge_cost_sub_array.reshape((5,25))
        
        # print("??????????????????????????????????????????")
        # print(f"agent_positions {agent_positions}")
        # print(f"adjM_with_edge_cost {adjM_with_edge_cost}")
        # print(f"adjM_with_reduced_edge_cost {adjM_with_reduced_edge_cost}")
        # print("??????????????????????????????????????????")
        
        # there are 3 types of reward design
        # 1.this reward is based on taking action on current state to next state
        # self.agent_rewards -= 1 #(step cost)
        # reward -=1
         
        # agents position updated after taking action
        print(f"old_agent_positions {self.agent_positions}")
        print(f"new_actions {actions}")
        new_agent_positions_after_action = self.take_action(actions)
        # update agent positions after taking action
        self.agent_positions = np.argmax(new_agent_positions_after_action, axis=1)
        
        # update agent positions after taking action
        print("new_agent_positions", self.agent_positions)
        print(f"agent new positions one hot after action {new_agent_positions_after_action}")
        old_agent_positions = np.argmax(agent_positions, axis=1)
        new_agent_positions = np.argmax(new_agent_positions_after_action, axis=1)
        
        #print(f"updated old_agent_positions {old_agent_positions} \nupdated new_agent_positions {new_agent_positions}")
        ###print(f"Orginal AdjM_with_redueced_edge_cost {self.adjM_with_reduced_cost}")
        
    
        
        edge_index_map = {}  # Dictionary to map edge index to number
        edge_number = 0 

        for i in range(len(adjM_with_edge_cost)):
            for j in range(len(adjM_with_edge_cost[i])):
                if i == j and j==0:
                    edge_number = 0
                else:
                    edge_number+= 1
                edge_index_map[(i, j)] = [edge_number, adjM_with_edge_cost[i][j]]
                # if i == edge_traversed[0] and j == edge_traversed[1]:
                #     edge_index_map[(i, j)] = edge_number
                #     break
        print("Edge Index Map:", edge_index_map) # edge: index+cost 
        self.edge_index_map = edge_index_map
        print(f"Adjm_with_reduced_edge_cost {adjM_with_reduced_edge_cost}")
        
        # 3 kinds if movement, 1) both move to new position,  2)both stay at same position (no reward or zero) 3) one move to new position while other stay at same position
        # if valid move -1 cost and if invalid move -100 cost in 1 and 2 case
        # if coordination move then +5 cost in 3 case else -1 cost cost if valid move and -100 if invalid move
        # for terminal state +10 reward
        
        #both stay at same position (no reward or zero)
        if old_agent_positions.all() == new_agent_positions.all():
            self.agent_rewards +=0
            reward +=0
        # both move to new position    
        elif old_agent_positions.all()!= new_agent_positions.all() and old_agent_positions[0] != new_agent_positions[0] and old_agent_positions[1] != new_agent_positions[1]:
            list_of_travelled_edges = []
            list_of_travelled_edges.append((old_agent_positions[0], new_agent_positions[0]))
            list_of_travelled_edges.append((old_agent_positions[1], new_agent_positions[1]))
            # check if the edges are valid or not, if valid move -1 if invalid move -100s
            print(f"list_of_travelled_edges {list_of_travelled_edges}")
            
            if list_of_travelled_edges[0] in edge_index_map and edge_index_map[(i, j)][1]!=0:
                self.agent_rewards -= 1
                reward -=1
                print(f"valid edge {list_of_travelled_edges[0]}")
            else:
                self.agent_rewards -= 100
                reward -=100 
                print(f"invalid edge {list_of_travelled_edges[0]}")
                
            if list_of_travelled_edges[1] in edge_index_map and edge_index_map[(i, j)][1]!=0:
                self.agent_rewards -= 1
                reward -=1
                print(f"valid edge {list_of_travelled_edges[1]}")
            else:
                self.agent_rewards -= 100
                reward -=100 
                print(f"invalid edge {list_of_travelled_edges[1]}")
                
      
        # To Do: this is optional (we assume agent will learn by themselvess for the moments)
        # coordination is possible only when one agent support other agent
        # if one agent stay and let other move to new position, then it is coordination
        elif old_agent_positions[0] == new_agent_positions[0] and old_agent_positions[1] != new_agent_positions[1] or old_agent_positions[0] != new_agent_positions[0] and old_agent_positions[1] == new_agent_positions[1]:
            edge_traversed = new_agent_positions
            if old_agent_positions[0] == new_agent_positions[0]:
                support_node = new_agent_positions[0]
            elif old_agent_positions[1] == new_agent_positions[1]:
                support_node = new_agent_positions[1]
            
            print(f"sopport node {support_node} \nedge traversed {edge_traversed}")
            
            ## check of the edge traversed is exists or not and find its index from adjM_with_edge_cost
         
            print(f"coordinated_edge {edge_traversed}")
            edge_traversed = tuple(edge_traversed)
            
            if edge_traversed in edge_index_map and edge_index_map[edge_traversed][1]==0:
                #time.sleep(50)
                self.agent_rewards -= 100 # invalid move
                reward -=100 
                print(f"invalid edge {edge_traversed}")
            else:
                print(f"valid edge {edge_traversed}")
                self.agent_rewards -= 1 # valid move
                reward -=1
                 
                
        #         ## only give reward when agent 1 support agent 2 and edge traversed is valid edge    
        #         for i in range(adjM_with_reduced_edge_cost.shape[0]):
        #             if i == support_node:
        #                 # self.agent_rewards += 500 
        #                 # reward +=500
        #                 edge_traversed_index = edge_index_map[(edge_traversed[0], edge_traversed[1])][0]
        #                 edge_cost_unredced = edge_index_map[(edge_traversed[0], edge_traversed[1])][1]
        #                 edge_cost_reduced = adjM_with_reduced_edge_cost[i][edge_traversed_index]
        #                 print(edge_index_map[(edge_traversed[0], edge_traversed[1])])
        #                 print(f"edge_traversed_index {edge_traversed_index}\nedge_cost_unredced {edge_cost_unredced}]\nedge_cost_reduced {edge_cost_reduced}")
                    
        #                 if  edge_cost_reduced == 5:
        #                     self.agent_rewards += 25  
        #                     reward +=25
        #                     print("Finally found the edge to reduce cost")
        #                 else:
        #                     self.agent_rewards -= 1
        #                     reward -=1
        #                     print("No edge cost is reduced")
                        
        # Should add penalty for moving into non-adjacent nodes
        # discuss with advisor

        # termination reward
        terminated = False
        info = {}
        if self.agent_positions[0] == 4 and self.agent_positions[1] == 4:
            self.agent_rewards+= 10
            reward += 10
            terminated = True
        
        return self._get_observation(),  reward,  terminated , info
       
        

    def reset(self):
        print(f"-------------------------Resetting the environment--------------------------")
        self.agent_positions = [0]*self.n_agents
        #one_hot_agent_positions = self.one_hot_encode_agent_positions()
        
        #print(f"agent positions {self.agent_positions} one hot {one_hot_agent_positions}")
        #return one_hot_agent_positions
        return self._get_observation()

    def render(self, mode='human'):
        plt.clf()
        nx.draw(self.graph.graph, pos=self.node_positions, with_labels=True, node_color='grey')

        # Overwrite the agent's node with a different color
        colors = ['blue', 'red', 'green', 'yellow', 'purple']  # extend this list if you have more than 5 agents
        for i, agent_position in enumerate(self.agent_positions):
            nx.draw_networkx_nodes(self.graph.graph, self.node_positions, nodelist=[agent_position], node_shape='*',node_color=colors[i], node_size=1000)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        self.frames.append(np.array(img))  # convert PIL Image to numpy array
        plt.close()

    def save_gif(self, filename):
        imageio.mimsave(filename, self.frames, format='GIF', fps=1)  # Save frames as a gif

    def save_mp4(self, filename):
        imageio.mimwrite(filename, self.frames, format='mp4', fps=1)  # Save frames as a mp4

    def close (self):
        pass

    def _get_observation(self):
        return self.get_observation_space()
        

    def _get_reward(self):
        # Reward function can be defined as per requirement
        return [0]*self.n_agents

    def _is_done(self):
        # Termination condition can be defined as per requirement
        #return [False]*self.n_agents
        
        if self.agent_positions[0] == 4 and self.agent_positions[1] == 4:
            return [True,True]
        else:
            return [False,False]
    

