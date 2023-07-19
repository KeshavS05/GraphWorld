import gym
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
import matplotlib.pyplot as plt

class GraphWorld():
    
    def __init__(self, parse_args=False, threeD=False, length=0, height=0, depth=2, adj=np.array([]), 
                 num_nodes=0, num_edges=0, fully_connected=True, 
                 n_agents=1, num_terminating_nodes=1, collaborative=False, 
                 ratio=0, weight_before=2, weight_after=0.5) -> None:
        
        self.parse_args = parse_args # Specify argument parsing
        self.threeD = threeD
        self.adj = adj # Adjacency matrix
        self.fully_connected = fully_connected # Specifies if every node should be reachable from every node
        self.n_agents = n_agents # Number of agents
        self.num_terminating_nodes = num_terminating_nodes
        self.collaborative = collaborative # Agents receive same reward
        self.G = nx.Graph() # Networkx graph
        self.non_terminal_nodes = [] # Non-terminal nodes (agents can move if on non-terminal node)
        self.terminal_nodes = [] # Terminal nodes (agents enter terminal state if they occupy terminal node)
        self.scale_length = 0 # Render scaling
        self.scale_height = 0 # Render scaling
        self.time_step = 0 # Environment's current time step
        self.agents = [] # List of agents
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
            self.weight_before = weight_before # Default weight of a weighted edge
            self.weight_after = weight_after # Weight of a weighted edge if agents occupy both nodes
        self.edges = np.array([]) # List of tuples (node1, node2, weight) describing edges in graph
        self.weighted_edges = np.array([]) # Weighted edges
        

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
        
    def create_adj(self):
        # Make adjacency matrix for the graph   
        if self.adj.size == 0:
            self.adj = self.random_adj(self.num_nodes, self.num_edges, self.fully_connected, self.ratio) 
        else:
            self.num_nodes = len(self.adj)
            if self.fully_connected:
                print("Warning: Graph may not be fully connected as it will adhere to the adjacency matrix provided.")
    
    def create_graph(self):
        # Create the graph 
        self.add_nodes()
        self.add_edges()
    
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


    def add_edges(self):
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                weight = self.adj[i][j]
                if weight != 0:
                    self.G.add_weighted_edges_from([(i, j, weight)])
                    # Supporting nodes are nodes with at least one weighted edge
                    if weight != 1:  
                        self.get_node(i).supporting = True
                        self.get_node(j).supporting = True
    
    
    def initalize_agents(self):
        # Initialize agents to random nodes in the graph (cannot start on a terminal node)
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(RandomAgent(i, self.get_node(idx := np.random.choice(self.non_terminal_nodes))))
            self.get_node(idx).num_agents += 1 


    def set_weights_of_edges(self):
        # Sets weights of edges
        for x, y, _ in self.weighted_edges:
            if self.get_node(x).num_agents > 0 and self.get_node(y).num_agents > 0: 
                # Change weight if there are agents on both of two connected support nodes (connected support nodes are two nodes connected by a weighted edge)
                self.G[x][y]["weight"] = self.weight_after 
            else:
                # Otherwise default weighted edge weight
                self.G[x][y]["weight"] = self.weight_before 

    def random_adj(self, num, num_edges, fully_connected, ratio):
        # Create random adjacency matrix if not provided by user
        # Adds only the number of edges specified by the user
        adjacent = np.zeros((num, num))
        possible = len(np.triu_indices(num, 1)[0])
        if fully_connected and num_edges < num - 1: # Warning for creating fully-connected graph without enough edges
            print("Warning: Graph will not be fully connected: Not enough edges")
        if num_edges > possible: # No isolated nodes allowed (every node must have at least one edge)
            print("Warning: The amount of edges exceeds those possible!")
            print(f"Edges reset to: {possible}")
            num_edges = possible
            
        edge_indicies = []

        if fully_connected: # Can be fully-connected if specified (every node is reachable from every node)
            rows = set()
            ind = 0
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
                adjacent[i][j] = 1
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
            adjacent[indicies[0][idx]][indicies[1][idx]] = 1
        
        num_of_weighted = int(ratio * len(edge_indicies)) # Ratio specifies number of edges that are weighted
        ind = 0
        weighted = set()
        while ind < num_of_weighted:
            tup = random.choice(edge_indicies)
            if tup not in weighted:
                weighted.add(tup)
                ind += 1
                adjacent[tup[0]][tup[1]] = self.weight_before # Sets these weighted edges values to weight_before
        adjacent = adjacent + adjacent.T
        print(adjacent)
        return adjacent


    def reset(self):
        # Create random adjacency matrix or use the one inputted
        self.create_adj()
        
        # Randomly sample terminal node labels
        self.terminal_nodes = random.sample(range(0, self.num_nodes), self.num_terminating_nodes)
        self.non_terminal_nodes = [i for i in range(self.num_nodes) if i not in self.terminal_nodes] 
        self.nodes = self.G.nodes()
        
        # Initialize the graph using networkx library based on user parameters
        self.create_graph()
              
        # Initialize the agents on random nodes   
        self.initalize_agents()
            
        # Create list of edges, and find edges that are weighted
        self.edges = self.G.edges(data="weight")
        self.weighted_edges = [i for i in self.edges if i[2] != 1]
        self.set_weights_of_edges()
        if self.threeD:
            self.R_3d = threeD_Render(self.length, self.height, self.depth, self.G, self.agents, self.weight_before, self.weight_after) # Render object for rendering 3d environment
        else:
            self.R = Render(self.length, self.height, self.G, self.agents, self.weight_before, self.weight_after) # Render object for rendering 2d environment
            self.scale_length = self.R.scale_length
            self.scale_height = self.R.scale_height
        #self.R = Render(self.length, self.height, self.G, self.agents, self.weight_before, self.weight_after)

        
        plt.ion()
    
    
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
            
    
    def get_obs(self, a):
        # Get agents observation
        return a.observation
        

    def step(self, actions=None):
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
                         
            rewards.append(self.get_reward(agent))
            observations.append(self.get_obs(agent))
            done += agent.node.terminating 
        self.time_step += 1
        
        if self.collaborative:
            rewards = [sum(rewards)] * self.n_agents
        
        self.set_weights_of_edges()
        done = (done == self.n_agents) 
        if done:
            plt.show()
        return observations, rewards, done 
            
    
    def render(self, image_size=800, grid_line_color=(87, 89, 93), grid_line_type=cv2.LINE_AA, grid_tick_val=1, default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 255, 255)):
        # Render all components of environment
        # Creates real-time matplotlib plot of environment
        # Returns BGR frame (3-dimensional numpy array)
        if not self.threeD:
            img_bgr, img_rgb = self.R.render(image_size, grid_line_color, grid_line_type, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, node_border_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color)
            if self.time_step == 0:
                self.im = plt.imshow(img_rgb)
            else:
                self.im.set_data(img_rgb)
            plt.show()
            plt.pause(1)

            return img_bgr
        else:
            self.R_3d.render(grid_line_color, grid_tick_val, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color)