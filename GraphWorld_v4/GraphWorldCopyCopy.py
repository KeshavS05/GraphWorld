import networkx as nx
import numpy as np
import random
import cv2
import time 
import argparse
import copy
from agents.random_agent import RandomAgent
from node import Node
from render import Render
from threeD_render import threeD_Render
from twoD_render import twoD_Render
import matplotlib.pyplot as plt
from subprocess import call
import copy
import itertools
from normalJSG import JSGraph
import pickle
from randomGraphGenerator import RandomGraph

class GraphWorld():
    
    def __init__(self, parse_args=False, html=True, threeD=False, length=0, height=0, depth=2, adj=np.array([]), type_of_weighted_matrix=np.array([]),
                 num_nodes=0, num_edges=0, fully_connected=True, 
                 n_agents=1, num_terminating_nodes=1, collaborative=False, 
                 ratio=0.2, default_weight=10, weight_before=10, weight_after=5, invalid_edge_weight=100, end_goals = np.array([]), start_goals = np.array([]),
                 JSG=False) -> None:
        
        self.parse_args = parse_args # Specify argument parsing
        self.html = html
        self.threeD = threeD
        self.adj = adj # Adjacency matrix
        self.type_of_weighted_matrix = type_of_weighted_matrix
        self.fully_connected = fully_connected # Specifies if every node should be reachable from every node
        self.n_agents = n_agents # Number of agents
        self.num_terminating_nodes = num_terminating_nodes
        self.collaborative = collaborative # Agents receive same reward
        self.JSG = JSG
        self.non_terminal_nodes = [] # Non-terminal nodes (agents can move if on non-terminal node)
        self.terminal_nodes = [] # Terminal nodes (agents enter terminal state if they occupy terminal node)
        self.scale_length = 0 # Render scaling
        self.scale_height = 0 # Render scaling
        self.time_step = 0 # Environment's current time step
        self.agents = [] # List of agents
        self.agent_rewards = 0
        if self.parse_args:
            print("Warning: Previous inputs instantiated in the environment's constructor will be overrided if given a new value in the arguments")
            self.read_arg_inputs()
        else:
            self.length = length # Length of positional grid
            self.height = height # Height of positional grid
            self.depth = depth
            self.num_nodes = num_nodes # Number of nodes possible in the graph
            self.num_edges = num_edges # Number of edges possible in the graph
            self.ratio = ratio # Ratio of weighted edges to normal edges
            self.default_weight = default_weight
            self.weight_before = weight_before # Default weight of a weighted edge
            self.weight_after = weight_after # Weight of a weighted edge if agents occupy both nodes
            self.invalid_edge_weight = invalid_edge_weight
        self.edges = np.array([]) # List of tuples (node1, node2, weight) describing edges in graph
        self.weighted_edges = np.array([]) # Weighted edges
        self.type_of_weighted = {}
        self.supporting_edge_dict = {}
        self.if_reset = False
        self.end_goals = end_goals
        self.start_goals = start_goals
        self.reset_num = 0
        if self.JSG:
            print('JSG Graph')
            # s = time.time()
            # V = self.num_nodes
            # rG = RandomGraph(V)         
            # rG.generateRandomGraph()        
            # E = rG.countEdgesEG()
            # risky_edges = int(self.ratio*E)
            # rG.generateRGWithSupportNodesAndRiskyEdges(risky_edges)
            # adj_ws = rG.getRGWithRiskEdgesAndSupportNodes() # EG with risky edges and support nodes
            # adj_ns = rG.getRGWithNoRiskEdgesAndSupportNodes() # EG   # goal posit
            # self.jsg = JSGraph(V)
            # self.jsg.graph = adj_ws
            # self.jsg.trans_Env_To_JSG() 
            # self.adj = self.jsg.get_adjMatrix_2D_JSG()
            # e = time.time()
            # self.original_adj = self.jsg.get_adjMatrix_2D_JSG()
            # #print("JSG Construction Time: "+ str(e-s))
        else:
            self.original_adj = np.copy(adj)
        self.G = nx.Graph() # Networkx graph
        

    def read_arg_inputs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--length', type=int, required=True)
        parser.add_argument('--height', type=int, required=True)
        parser.add_argument('--depth', type=int)
        parser.add_argument('--adj', type=int, nargs='+', action='append')
        parser.add_argument('--num_nodes', type=int, required=True)
        parser.add_argument('--num_edges', type=int, required=True)
        parser.add_argument('--fully_connected', type=bool)
        parser.add_argument('--n_agents', type=int)
        parser.add_argument('--num_terminating_nodes', type=int)
        parser.add_argument('--collaborative', type=bool)
        parser.add_argument('--ratio', type=float, required=True)
        parser.add_argument('--weight_before', type=float, required=True)
        parser.add_argument('--weight_after', type=float, required=True)
        args = parser.parse_args()
        self.length = args.length
        self.height = args.height
        self.num_nodes = args.num_nodes
        self.num_edges = args.num_edges
        self.ratio = args.ratio
        self.weight_before = args.weight_before
        self.weight_after = args.weight_after
        if args.depth: self.depth = args.depth
        if args.adj: self.adj = np.array(args.adj)
        if args.fully_connected: self.fully_connected = args.fully_connected 
        if args.n_agents: self.n_agents = args.n_agents
        if args.num_terminating_nodes: self.num_terminating_nodes = args.num_terminating_nodes 
        if args.collaborative: self.collaborative = args.collaborative 


    def get_node(self, node_label):
        # Get a node based on its unique node label
        return self.nodes[node_label]["val"]
    
    def get_edge(self, node_one, node_two, val):
        # Get an edge between 2 nodes in a graph
        return self.G[node_one][node_two][val]
    
    def get_neighbors(self, node):
        # Get neighbors of a node in the graph
        return list(self.G[node.node_label].keys())
    
    def q_get_neighbors(self, node):
        return list(self.G.neighbors(node))
        
    def create_adj(self, random_default_edges):
        # Make adjacency matrix for the graph   
        if len(self.adj) == 0:
            # Create random adj matrix
            self.adj = self.random_adj(self.num_nodes, self.num_edges, self.fully_connected, self.ratio, random_default_edges)    
        else:   
            self.num_nodes = len(self.adj)
            if not len(self.type_of_weighted_matrix):
                if self.reset_num== 0: print("Warning: Type of edge (0: Default, 1: Weight Before, 2: Weight After) not inputted with adjacency matrix")
            for i in range(len(self.type_of_weighted_matrix)):
                for j in range(len(self.type_of_weighted_matrix)):
                    t = self.type_of_weighted_matrix[i][j]
                    if t not in [0, 1, 2]:
                        continue
                    self.type_of_weighted[(i, j)] = t
            if self.fully_connected:
                if self.reset_num== 0: print("Warning: Graph may not be fully connected as it will adhere to the adjacency matrix provided.")
    
    def create_graph(self, random_adj):
        # Create the graph 
        self.add_nodes()
        self.add_edges(random_adj)
    def add_nodes(self):
        node_label = 0
        coords_occupied = set()

        # Add number of nodes specified by user 
        while self.G.number_of_nodes() < self.num_nodes:
            # Specify if node is terminal or not
            if node_label in self.terminal_nodes:
                node = Node(node_label, self.length, self.height, self.depth, True, supporting=False)
            else:
                node = Node(node_label, self.length, self.height, self.depth, False, supporting=False)
            if node.return_coords() not in coords_occupied: # If node's position is not occupied
                self.G.add_node(node_label, val=node)
                coords_occupied.add(node.return_coords())
                node_label += 1


    def add_edges(self, random_adj):
        #self.s_adj = np.zeros((len(self.adj), len(self.adj)))
        if random_adj or (not self.if_reset and len(self.supporting_edge_dict) == 0):
            self.s_adj = [[0 for _ in range(len(self.adj))] for _ in range(len(self.adj))]
            for i in range(self.num_nodes):
                for j in range(i, self.num_nodes):
                    weight = self.adj[i][j]
                    if i == j:
                        continue
                    elif (i, j) not in self.type_of_weighted:
                        self.s_adj[i][j] = self.invalid_edge_weight
                        self.s_adj[j][i] = self.invalid_edge_weight
                    else:
                        self.G.add_weighted_edges_from([(i, j, weight)])
                        # Supporting nodes are nodes with at least one weighted edge
                        if self.type_of_weighted[(i, j)] == 0:#weight == 1:
                            self.s_adj[i][j] = self.default_weight
                            self.s_adj[j][i] = self.default_weight
                        else:  
                            ind1 = np.random.randint(0, self.num_nodes)
                            while self.get_node(ind1).terminating:
                                ind1 = np.random.randint(0, self.num_nodes)
                            ind2 = np.random.randint(0, self.num_nodes)
                            while ind2 == ind1 or self.get_node(ind2).terminating:
                                ind2 = np.random.randint(0, self.num_nodes)
                            self.get_node(ind1).supporting = True
                            self.get_node(ind2).supporting = True

                            self.supporting_edge_dict[(i, j)] = (ind1, ind2)
                            self.s_adj[i][j] = [weight, [ind1, ind2]]
                            self.s_adj[j][i] = [weight, [ind2, ind1]]
                            
        else:
            for i in range(self.num_nodes):
                for j in range(i, self.num_nodes):
                    weight = self.adj[i][j]
                    if i == j:
                        continue
                    elif (i, j) not in self.type_of_weighted:
                        continue
                    else:
                        self.G.add_weighted_edges_from([(i, j, weight)])
                        # Supporting nodes are nodes with at least one weighted edge
                        if self.type_of_weighted[(i, j)] == 0:#weight == 1:
                            continue
                        else:
                            for ind in self.supporting_edge_dict[(i, j)]:
                                self.get_node(ind).supporting = True
                            # ind1, ind2 = self.supporting_edge_dict[(i, j)]
                            # self.get_node(ind1).supporting = True
                            # self.get_node(ind2).supporting = True
    
    
    def initalize_agents(self, random_agents):
        # Initialize agents to random nodes in the graph (cannot start on a terminal node)
        self.agents = []
        self.agent_positions = []
        self.original_agents = []
        self.original_agent_positions = []
        if random_agents or not len(self.start_goals) > 0:
            for i in range(self.n_agents):
                self.agents.append(RandomAgent(i, self.get_node(idx := np.random.choice(self.non_terminal_nodes))))
                self.agent_positions.append(idx)
                self.original_agents.append(self.agents[i])
                self.original_agent_positions.append(self.agent_positions[i])
                self.get_node(idx).num_agents += 1 
        else:
            for i in range(self.n_agents):
                self.agents.append(RandomAgent(i, self.get_node(self.start_goals[i])))
                self.agent_positions.append(self.start_goals[i])
                self.original_agents.append(self.agents[i])
                self.original_agent_positions.append(self.agent_positions[i])
                self.get_node(self.start_goals[i]).num_agents += 1 

    def set_weights_of_edges(self):
        # Sets weights of edges
        for x, y in self.supporting_edge_dict:
            #ind1, ind2 = self.supporting_edge_dict[(x, y)]
            b = False
            for ind in self.supporting_edge_dict[(x, y)]:
                if self.get_node(ind).num_agents > 0:
                    b = True
                    break
            if b:
                # Change weight if there are agents on both of two connected support nodes (connected support nodes are two nodes connected by a weighted edge)
                self.G[x][y]["weight"] = self.weight_after   
                self.adj[x][y] = self.weight_after
                self.adj[y][x] = self.weight_after
                self.type_of_weighted[(x, y)] = 2
                self.type_of_weighted[(y, x)] = 2
            else:
                # Otherwise default weighted edge weight
                self.G[x][y]["weight"] = self.weight_before   
                self.adj[x][y] = self.weight_before
                self.adj[y][x] = self.weight_before
                self.type_of_weighted[(x, y)] = 1
                self.type_of_weighted[(y, x)] = 1

    def random_adj(self, num, num_edges, fully_connected, ratio, random_default_edges):
        # Create random adjacency matrix if not provided by user
        # Adds only the number of edges specified by the user
        #adjacent = np.zeros((num, num))
        
        adjacent = np.array([[0 for i in range(num)] for j in range(num)])
        for i in range(num):
            for j in range(i + 1, num):
                adjacent[i][j] = self.invalid_edge_weight
                
        # print("invalid: ", adjacent)
        possible = len(np.triu_indices(num, 1)[0])
        if fully_connected and num_edges < num - 1: # Warning for creating fully-connected graph without enough edges
            if self.reset_num== 0: print("Warning: Graph will not be fully connected: Not enough edges")
        if num_edges > possible: # No isolated nodes allowed (every node must have at least one edge)
            if self.reset_num== 0: print("Warning: The amount of edges exceeds those possible!")
            if self.reset_num== 0: print(f"Edges reset to: {possible}")
            num_edges = possible
            
        edge_indicies = []
        self.type_of_weighted = {}
        ind = 0
        if random_default_edges and self.supporting_edge_dict:
            for x, y in self.supporting_edge_dict:
                if ind >= num_edges:
                    break
                if adjacent[x][y] == self.invalid_edge_weight:
                    adjacent[x][y] = self.weight_before  
                elif adjacent[y][x] == self.invalid_edge_weight:
                    adjacent[y][x] = self.weight_before  
                self.type_of_weighted[(x, y)] = 1
                self.type_of_weighted[(y, x)] = 1
                edge_indicies.append((x, y))
                ind += 1 
        # print("weighted before: ", adjacent)
        if fully_connected: # Can be fully-connected if specified (every node is reachable from every node)
            rows = set()
            while ind < (num - 1):
                if ind >= num_edges:
                    break
                i = np.random.randint(0, num - 1)
                while i in rows:
                    i = np.random.randint(0, num - 1)
                rows.add(i)
                j = np.random.randint(i + 1, num)
                while j in rows:
                    j = np.random.randint(i + 1, num)  
                if (i, j) in edge_indicies:
                    continue
                adjacent[i][j] = self.default_weight
                self.type_of_weighted[(i, j)] = 0
                self.type_of_weighted[(j, i)] = 0
                edge_indicies.append((i, j))
                ind += 1
        else:
            ind = 0
        indicies = np.triu_indices(num, 1)
        for i in range(ind, num_edges):
            idx = np.random.randint(0, possible)
            while (indicies[0][idx], indicies[1][idx]) in edge_indicies:
                idx = np.random.randint(0, possible)
            edge_indicies.append((indicies[0][idx], indicies[1][idx]))
            adjacent[indicies[0][idx]][indicies[1][idx]] = self.default_weight
            self.type_of_weighted[(indicies[0][idx], indicies[1][idx])] = 0
            self.type_of_weighted[(indicies[1][idx], indicies[0][idx])] = 0
        if not random_default_edges or not self.supporting_edge_dict:
            num_of_weighted = int(ratio * len(edge_indicies)) # Ratio specifies number of edges that are weighted
            ind = 0
            weighted = set()
            while ind < num_of_weighted:
                tup = random.choice(edge_indicies)
                if tup not in weighted:
                    weighted.add(tup)
                    ind += 1
                    adjacent[tup[0]][tup[1]] = self.weight_before # Sets these weighted edges values to weight_before
                    self.type_of_weighted[tup] = 1
                    self.type_of_weighted[(tup[1], tup[0])] = 1
        adjacent = adjacent + adjacent.T
        # print(adjacent)
        self.original_adj = np.copy(adjacent)
        return adjacent

    def adj_to_str(self):
        s = ''
        for l in self.adj:
            for n in l:
                s += str(n)
                s += '   '
            s += '\n'
        return s
    
    def reset(self, random_default_edges=False, random_adj=False, random_edge=False, random_agents=False, random_terminal = False, adj_input = np.array([])):
        self.time_step = 0
        self.G = nx.Graph()
        if len(adj_input) > 0:
            self.adj = adj_input
        elif random_adj or random_default_edges:
            self.adj = np.array([])   
        else:
            self.adj = np.copy(self.original_adj) 
        # Create random adjacency matrix or use the one inputted
        if random_adj or not self.supporting_edge_dict:
            self.supporting_edge_dict = {}
        self.nodes = self.G.nodes()
        self.create_adj(random_default_edges)
        if random_adj or not self.if_reset or random_terminal:
            # Randomly sample terminal node labels
            self.original_adj = np.copy(self.adj)
            if len(self.end_goals) > 0:
                self.terminal_nodes = self.end_goals
                self.non_terminal_nodes = [i for i in range(self.num_nodes) if i not in self.terminal_nodes]
            else:
                self.terminal_nodes = random.sample(range(0, self.num_nodes), self.num_terminating_nodes)
                self.non_terminal_nodes = [i for i in range(self.num_nodes) if i not in self.terminal_nodes] 
        self.create_graph(random_adj=random_adj)
        
        # Initialize the agents on random nodes 
        if not self.if_reset or random_adj:
            if random_agents:
                self.initalize_agents(random_agents=True)
            else:
                self.initalize_agents(random_agents=False)
        else:
            for i in range(len(self.original_agent_positions)):
                self.agents[i] = RandomAgent(i, self.get_node(self.original_agent_positions[i]))
                self.get_node(self.original_agent_positions[i]).num_agents += 1 
                
            self.agent_positions = self.original_agent_positions

        self.set_weights_of_edges()
        if self.threeD:
            self.R_3d = threeD_Render(self.length, self.height, self.depth, self.G, self.agents, self.type_of_weighted, self.supporting_edge_dict) # Render object for rendering 3d environment
        else:
            if self.html:
                self.R_2d = twoD_Render(self.html, self.length, self.height, self.depth, self.G, self.agents, self.type_of_weighted, self.time_step)
            else:
                self.R_2d = Render(self.length, self.height, self.G, self.agents, self.type_of_weighted)
        self.if_reset = True
        self.reset_num += 1
        return self._get_observation() 
    

    def get_random_action(self, a):
        # Return random action based on the neighbors of the node the agent is currently on
        return np.random.choice([a.node.node_label] + self.get_neighbors(a.node))

    
    def get_reward(self, a):
        # Djikstra-based distance from terminal reward
        shortest_path_to_terminal = self.num_nodes 
        for terminal_node in self.terminal_nodes:
            if (l := len(nx.shortest_path(self.G, source = a.node.node_label, target = terminal_node, weight = "weight"))) < shortest_path_to_terminal:
                shortest_path_to_terminal = l
        return shortest_path_to_terminal       
    
    def step_random(self, actions=None):
        # Step one time-step in the environment
        rewards = []
        observations = []
        done = 0   
        for i, agent in enumerate(self.agents):
            if actions: 
                action = actions[i]
            else: 
                action = self.get_random_action(agent)

            if not agent.node.terminating: 
                agent.node.num_agents -= 1
                agent.node = self.get_node(action)
                agent.node.num_agents += 1
                         
            #rewards.append(self.get_reward(agent))
            #observations.append(self.get_obs(agent))
            done += agent.node.terminating 
        self.time_step += 1
        self.set_weights_of_edges()
        if self.collaborative:
            rewards = [sum(rewards)] * self.n_agents       
        return 0, 0, done == self.n_agents#observations, rewards, done == self.n_agents
    
    def render(self, image_size=800, grid_line_color=(255, 255, 255), grid_line_thickness=1, grid_tick_val=1, grid_line_type=cv2.LINE_AA, default_edge_color=(0, 255, 0), active_support_edge_color=(255, 174, 66), inactive_support_edge_color=(255, 255, 255), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 255, 255), agent_color=(255, 0, 0), agent_label_text_color=(255, 255, 255)):
        # Render all components of environment
        # Creates real-time matplotlib plot of environment
        # Returns BGR frame (3-dimensional numpy array)
        if not self.threeD:
            if self.html:
                return self.R_2d.render(image_size, grid_line_color, grid_tick_val, grid_line_type, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, node_border_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color)
            else:
                return self.R_2d.render(image_size, grid_line_color, grid_line_thickness, grid_line_type, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, node_border_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color)
        else:
            return self.R_3d.render(grid_line_color, grid_tick_val, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color)
        
    def get_random_node_from_list(self, nodes):
        random_index = random.randint(0, len(nodes) - 1)
        random_node = nodes[random_index]
        return random_node   
    
    def get_actions(self, observation):
        
        pair_of_actions = []
        for agent_obs in observation:
            list_of_actions = self.get_action_space(agent_obs) # single agent
            action = self.get_random_node_from_list(list_of_actions)
            pair_of_actions.append(action)
        return pair_of_actions
    
    def normalize_reward(self, reward):
        #print(reward)
        min_reward = -1001 * self.n_agents
        max_reward = 100
        normalized_reward = (reward - min_reward) / (max_reward - min_reward)
        #print(normalized_reward)
        return reward

    # tesing phase (worked like charm so far)
    def step(self, actions, print_flag=False):
        wall_penalty=0
        # this works for pairwise actions
        ### valid actions
        if self.n_agents == sum(1 if actions[n] in self.get_action_space(self.agent_positions[n]) else 0 for n in range(self.n_agents)):
            moving_labels = [] 
            staying_labels = []
            for n in range(self.n_agents): 
                if self.agent_positions[n] != actions[n]:
                    moving_labels.append(n)
                else:
                    staying_labels.append(n)
          
            b = False
            if len(moving_labels) == 1: 
                #time.sleep(2)
                support_node = [actions[i] for i in staying_labels]
                risky_edge = tuple(sorted((self.agent_positions[moving_labels[0]], actions[moving_labels[0]])))
                #risky_edge = (self.agent_positions[1], actions[1])
                # print(f"support_node: {support_node} risky_edge: {risky_edge} risky_edge_with_support_nodes: {self.graph.risk_edges_with_support_nodes}")
                if risky_edge in self.supporting_edge_dict:
                    for node in support_node:
                        if node in self.supporting_edge_dict[risky_edge]:
                            reward = 2
                            break
                        elif node not in self.supporting_edge_dict[risky_edge]:
                            b = True
                            reward = -5 # very high cost of moving without support node
                        
                else: 
                    reward = 1 * self.n_agents # step cost       
            
            elif 1 < len(moving_labels) < (self.n_agents - 1):
                support_node = [actions[i] for i in staying_labels]
                risky_edges = [tuple(sorted((self.agent_positions[i], actions[i]))) for i in moving_labels]
                # print(f"support_node: {support_node} risky_edge: {risky_edge} risky_edge_with_support_nodes: {self.graph.risk_edges_with_support_nodes}")
                reward = 0
                n = 0
                for risky_edge in risky_edges:
                    if risky_edge in self.supporting_edge_dict:
                        n += 1
                        b2 = True
                        for node in support_node:
                            if node in self.supporting_edge_dict[risky_edge]:
                                reward += 2
                                b2 = False
                                break

                        if b2: 
                            reward -= 5    
                    else: 
                        reward += 1 # step cost  
                if n >= 2:
                    reward = -3

            # All agents except for one move         
            elif len(moving_labels) == (self.n_agents - 1):
                #time.sleep(2)
                support_node = actions[staying_labels[0]]
                #risky_edge = (self.agent_positions[moving_labels[0]], actions[moving_labels[0]])
                risky_edges = [tuple(sorted((self.agent_positions[i], actions[i]))) for i in moving_labels]
                # print(f"support_node: {support_node} risky_edge: {risky_edge} risky_edge_with_support_nodes: {self.graph.risk_edges_with_support_nodes}")
                reward = 0
                n = 0
                for risky_edge in risky_edges:
                    if risky_edge in self.supporting_edge_dict:
                        n += 1
                        if support_node in self.supporting_edge_dict[risky_edge]:
                            reward += 2
                        elif support_node not in self.supporting_edge_dict[risky_edge]:
                            b = True
                            reward -= 5 # very high cost of moving without support node
                            
                    else: 
                        reward += 1 # step cost  
                if n >= 2:
                    reward = -3
            # if both moving or staying at same place 
            else:
                # print(">>>>>>>both moving or staying at same place<<<<<<<<<")
                reward = 0
                ## both moving
                if len(moving_labels) == self.n_agents:
                    traverse = [tuple(sorted((self.agent_positions[i], actions[i]))) for i in range(self.n_agents)]
                    rew = [x for x in traverse if x in self.supporting_edge_dict]
                    reward += ((len(rew) * -5) + ((self.n_agents - len(rew)) * 1))
                    if len(rew) >= 1:
                        b = True    
                else:
                    # both staying at same place
                    reward = -5 * self.n_agents#-22 # step cost reduction
                
            self.agent_positions = actions
            for ind, a in enumerate(self.agents):
                a.node.num_agents -=1 
                a.node = self.get_node(self.agent_positions[ind])
                a.node.num_agents += 1
        ## invalid actions
        else:
            wall_penalty = -5
            # hitting a wall concept dont need to update agent positions
          
            
        epsilon = - 0.05 #(closer to goal, higher reward, less negative value)
        total_distance_to_goal = 0
        for agent_id in range(self.n_agents):
                shortest_path_length = self.shortest_path_length(
                    self.agent_positions[agent_id], self.end_goals[agent_id]
                )
                total_distance_to_goal += shortest_path_length

        avg_distance_to_goal = total_distance_to_goal / self.n_agents
        distance_to_goal_reward =  epsilon * avg_distance_to_goal

        # print("avg_distace_to_goal", avg_distance_to_goal)
        # print("distance_to_goal_reward", distance_to_goal_reward)
        # print("coordination_reward", reward)
        
        # closer to goal, higher reward, more positive value
        coordination_factor = 1.2**(-avg_distance_to_goal)
        # print("coordination_factor", coordination_factor)
        # time.sleep(1)

        ## wall/stagnant penalty plus no coordination
        observation = self.agent_positions
        if self.n_agents == sum(1 if self.get_node(self.agent_positions[i]).terminating else 0 for i in range(self.n_agents)):
            done = True
            if b:
                goal_reward = reward + 10
            else:
                goal_reward = 10
            step_cost = 0
        else:
            done = False
            goal_reward = 0
            step_cost = -0.01
        reward = step_cost + wall_penalty + reward*(-distance_to_goal_reward) + distance_to_goal_reward + goal_reward
        self.time_step += 1
        
        # reward = self.normalize_reward(reward)
        return observation, reward, done

    def close (self):
        pass
    

    def _get_observation(self):
        obs =  self.agent_positions
        return obs

    def _get_reward(self):
        # Reward function can be defined as per requirement
        
        #if self.agent_positions[0] == self.graph.end_goal and self.agent_positions[1] == self.graph.end_goal:
        if sum(1 if a in self.terminal_nodes else 0 for a in self.agent_positions) == len(self.agent_positions):#sum(1 if a.node.terminating else 0 for a in self.agents) == self.n_agents:
            return 100
        else:
            return -1

    def _is_done(self):
        # Termination condition can be defined as per requirement (time factor or goal based)
        #if self.agent_positions[0] == self.graph.end_goal and self.agent_positions[1] == self.graph.end_goal:
        if sum(1 if a in self.terminal_nodes else 0 for a in self.agent_positions) == len(self.agent_positions):#sum(1 if a.node.terminating else 0 for a in self.agents) == self.n_agents:
            return True
        else:
            return False
    
    def get_action_space(self, agent_position):
        #agent = self.agents[agent_position]
        action_space = [neighbour for neighbour in self.get_neighbors(self.get_node(agent_position))]#self.graph.get_neighbors(agent_position)]
        action_space.append(agent_position)
        return action_space
    
    def get_neighbors_new(self, node):
        return list(self.G.neighbors(node))
    
    def init_testing_params(self):
        n_actions = self.n_agents
        nodes = self.G.nodes
        states_pairs = list(itertools.product(nodes, repeat=n_actions))
        states_value_index = {state: index for index, state in enumerate(states_pairs)}
        
        
        # Load Q values from the pickle file
        with open('q_values_risk_large.pkl', 'rb') as f:
            q_values = pickle.load(f)
            
        self.q = q_values
        self.states_pairs = states_pairs
        self.states_values_index = states_value_index
    
    def step_test(self, state):
        state_index = self.states_values_index[tuple(state[i] for i in range(self.n_agents))]
        q_values = self.q[state_index,:]
        action_index = np.argmax(q_values)
        
        action = self.states_pairs[action_index]
        next_state, reward, done = self.step(action, print_flag=True)
        self.set_weights_of_edges()
        return next_state, reward, done
    
    
    def input_adj(self, edges, risk_edges):
        if self.JSG:
            #adj = np.full((self.num_nodes, self.num_nodes), 0, dtype = float)
            adj = [[0 for i in range(self.num_nodes)] for j in range(self.num_nodes)]
            type_of_weighted = np.full((self.num_nodes, self.num_nodes), None)
            for x, y in edges:
                if (x, y) in risk_edges:
                    adj[x][y] = [self.weight_before, risk_edges[(x, y)]]
                    adj[y][x] = [self.weight_before, risk_edges[(x, y)]]
                    type_of_weighted[x][y] = 1
                    type_of_weighted[y][x] = 1
                else:
                    adj[x][y] = [self.default_weight, tuple()]
                    adj[y][x] = [self.default_weight, tuple()]
                    type_of_weighted[x][y] = 0
                    type_of_weighted[y][x] = 0
            self.type_of_weighted_matrix = type_of_weighted
            self.supporting_edge_dict = risk_edges
            self.adj = adj
            # s = time.time()
            # self.jsg = JSGraph(self.num_nodes)
            # self.jsg.graph_1d = adj
            # self.jsg.trans_Env_To_JSG() 
            # self.adj = self.jsg.get_adjMatrix_2D_JSG()
            # self.num_nodes = len(self.adj)
            # new_adj = np.zeros((self.num_nodes, self.num_nodes))
            # for i in range(len(self.adj)):
            #     for j in range(len(self.adj[0])):
            #         val = self.adj[i][j]
            #         if val == 0 or val == float('inf') or i == j:
            #             new_adj[i][j] = 0
            #         else:
            #             print(i, j, val)
            #             supporting_nodes = set()
            #             for l in val:
            #                 if l[1]:
            #                     for x in l[1]:
            #                         supporting_nodes.add(x)
            #             if val[1]:
            #                 new_adj[i][j] = self.weight_before
            #                 self.type_of_weighted[(i, j)] = 1
            #                 self.supporting_edge_dict[(i, j)] = tuple(supporting_nodes)
            #                 #self.supporting_edge_dict[(i, j)] = val[1]
            #             else:
            #                 new_adj[i][j] = self.default_weight
            #                 self.type_of_weighted[(i, j)] = 0
            # print(self.supporting_edge_dict)
            # self.adj = np.copy(new_adj)
            # self.original_adj = np.copy(new_adj)
            # e = time.time()
            # print("JSG Construction Time: "+ str(e-s))
        else:
            adj = np.full((self.num_nodes, self.num_nodes), 0, dtype = float)
            type_of_weighted = np.full((self.num_nodes, self.num_nodes), None)
            for x, y in edges:
                if (x, y) in risk_edges:
                    adj[x][y] = self.weight_before
                    adj[y][x] = self.weight_before
                    type_of_weighted[x][y] = 1
                    type_of_weighted[y][x] = 1
                else:
                    adj[x][y] = self.default_weight
                    adj[y][x] = self.default_weight
                    type_of_weighted[x][y] = 0
                    type_of_weighted[y][x] = 0
            self.adj = adj
            self.original_adj = np.copy(adj)
            self.type_of_weighted_matrix = type_of_weighted
            self.supporting_edge_dict = risk_edges
        
    def shortest_path(self, start, goal):
        return nx.shortest_path(self.G, start, goal)
    
    def shortest_path_length(self, start, goal):
        return nx.shortest_path_length(self.G, start, goal)