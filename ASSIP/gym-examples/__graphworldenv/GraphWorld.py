import gym
import networkx as nx
import numpy as np
import random
import cv2
import time 
import argparse
from .agents import GraphWorldDQNAgent
import DQNAgent

class RandomAgent():

    def __init__(self, label, node):
        self.label = label
        self.node = node
        self.observation = {"Label": self.label, "Coords" : self.node.return_coords()}       
    
class Node(): 

    def __init__(self, node_label, length, height, terminating) -> None:
        self.node_label = node_label
        self.x = random.randint(1, length - 1)
        self.y = random.randint(1, height - 1)
        self.terminating = terminating
    
    def return_coords(self):
        return (self.x, self.y)
    
    def scale_coords(self, scale_length, scale_height):
        tup = self.return_coords()
        x = tup[0] * scale_length
        y = tup[1] * scale_height
        return (x, y)
        
    def equals(self, node):
        return self.return_coords == node.return_coords
    

class GraphWorld():
    
    def __init__(self, parse_args=False, length=0, height=0, adj=np.array([]), 
                 num_nodes=0, num_edges=0, fully_connected=True, 
                 n_agents=1, agent_start_locations=0, 
                 num_terminating_nodes=1, collaborative=False, ratio = 0, adv = 0) -> None:
        
        self.parse_args = parse_args
        self.adj = adj
        self.fully_connected = fully_connected
        self.n_agents = n_agents
        self.agent_start_locations = agent_start_locations
        self.num_terminating_nodes = num_terminating_nodes
        self.collaborative = collaborative
        self.G = nx.Graph()
        self.non_terminal_nodes = []
        self.terminal_nodes = []
        if self.parse_args:
            print("Warning: Previous inputs instantiated in the environment's constructor will be overrided if given a new value in the arguments")
            self.read_arg_inputs()
        else:
            self.length = length
            self.height = height
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.ratio = ratio
            self.adv = adv
        self.reset()
        

    def read_arg_inputs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--length', type=int, required=True)
        parser.add_argument('--height', type=int, required=True)
        parser.add_argument('--adj', type=int, nargs='+', action='append')
        parser.add_argument('--num_nodes', type=int, required=True)
        parser.add_argument('--num_edges', type=int, required=True)
        parser.add_argument('--fully_connected', type=bool)
        parser.add_argument('--n_agents', type=int)
        parser.add_argument('--agent_start_locations', type=int)
        parser.add_argument('--num_terminating_nodes', type=int)
        parser.add_argument('--collaborative', type=bool)
        parser.add_argument('--ratio', type=float, required=True)
        parser.add_argument('--adv', type=float, required=True)
        args = parser.parse_args()
        self.length = args.length
        self.height = args.height
        self.num_nodes = args.num_nodes
        self.num_edges = args.num_edges
        self.ratio = args.ratio
        self.adv = args.adv
        if args.adj: self.adj = np.array(args.adj)
        if args.fully_connected: self.fully_connected = args.fully_connected
        if args.n_agents: self.n_agents = args.n_agents
        if args.agent_start_locations: self.agent_start_locations = args.agent_start_locations
        if args.num_terminating_nodes: self.num_terminating_nodes = args.num_terminating_nodes
        if args.collaborative: self.collaborative = args.collaborative


    def reset(self):
        if adj.size == 0:
            self.adj = self.random_adj(self.num_nodes, self.num_edges, self.fully_connected, self.ratio, self.adv)
        else:
            self.adj = adj
            self.num_nodes = len(self.adj)
            if self.fully_connected:
                print("Warning: Graph may not be fully connected as it will adhere to the adjacency matrix provided.")
        
        node_label = 0
        coords_occupied = set()
        terminating_node_indices = random.sample(range(0, self.num_nodes - 1), self.num_terminating_nodes)
        while self.G.number_of_nodes() < self.num_nodes:
            if node_label in terminating_node_indices:
                node = Node(node_label, self.length, self.height, True)
            else:
                node = Node(node_label, self.length, self.height, False)
            if node.return_coords() not in coords_occupied:
                if node.terminating:
                    self.terminal_nodes.append(node)
                else:
                    self.non_terminal_nodes.append(node)
                self.G.add_node(node_label, val = node, coord = node.return_coords())
                coords_occupied.add(node.return_coords())
                node_label += 1

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                weight = self.adj[i][j]
                self.G.add_weighted_edges_from([(i, j, weight)])

        self.nodes = self.G.nodes()
        if self.agent_start_locations:
            self.agents = [RandomAgent(i, (n := self.nodes[self.agent_start_locations]["val"])) if n not in self.terminal_nodes else RandomAgent(i, random.choice(self.non_terminal_nodes)) for i in range(self.n_agents)]
        else:
            self.agents = [RandomAgent(i, random.choice(self.non_terminal_nodes)) for i in range(self.n_agents)]     

    
    def random_adj(self, num, num_edges, fully_connected, ratio, adv):
        adjacent = np.zeros((num, num))
        possible = len(np.triu_indices(num, 1)[0])
        if fully_connected and num_edges < num - 1:
            print("Warning: Graph will not be fully connected: Not enough edges")
        if num_edges > possible:
            print("Warning: The amount of edges exceeds those possible!")
            print(f"Edges reset to: {possible}")
            num_edges = possible
            
        edge_indicies = []

        if fully_connected:
            rows = set()
            ind = 0
            while ind < (num - 1):
                if ind >= num_edges:
                    break
                i = random.randint(0, num - 2)
                while i in rows:
                    i = random.randint(0, num - 2)
                rows.add(i)
                j = random.randint(i + 1, num - 1)
                while j in rows:
                    j = random.randint(i + 1, num - 1)  
                adjacent[i][j] = 1
                edge_indicies.append((i, j))
                ind += 1
        else:
            ind = 0
        indicies = np.triu_indices(num, 1)
        for i in range(ind, num_edges):
            idx = random.randint(0, possible - 1)
            while (indicies[0][idx], indicies[1][idx]) in edge_indicies:
                idx = random.randint(0, possible - 1)
            edge_indicies.append((indicies[0][idx], indicies[1][idx]))
            adjacent[indicies[0][idx]][indicies[1][idx]] = 1
        
        num_of_weighted = int(ratio * len(edge_indicies))
        ind = 0
        weighted = set()
        while ind < num_of_weighted:
            tup = random.choice(edge_indicies)
            if tup not in weighted:
                weighted.add(tup)
                ind += 1
                adjacent[tup[0]][tup[1]] = adv
        adjacent = adjacent + adjacent.T
        print(edge_indicies)
        print(adjacent)
        return adjacent

    
    def get_random_action(self, a):
        return random.choice(list(self.G[a.node.node_label].keys()))

    
    def get_reward(self, a):
        shortest_path_to_terminal = self.num_nodes + 1000
        for terminal_node in self.terminal_nodes:
            if (l := len(nx.shortest_path(self.G, a.node.node_label, terminal_node.node_label))) < shortest_path_to_terminal:
                shortest_path_to_terminal = l
        return shortest_path_to_terminal       
            
    
    def get_obs(self, a):
        return a.observation
        

    def step(self):
        rewards = []
        observations = []
        done = 0
        for agent in self.agents:
            action = self.get_random_action(agent)
            if not agent.node.terminating:
                agent.node = self.nodes[action]["val"]
                agent.observation["Coords"] = agent.node.return_coords()           
            rewards.append(self.get_reward(agent))
            observations.append(self.get_obs(agent))
            done += agent.node.terminating 
        
        if self.collaborative:
            rewards = [sum(rewards)/self.n_agents] * self.n_agents
        
        return observations, rewards, done == self.n_agents
            

    def draw_grid(self, img, scale_length, scale_height, line_color, thickness, type):
        pxstep = scale_length
        pystep = scale_height
        x = scale_length
        y = scale_height
        while x < (self.length * scale_length - scale_length + 1):
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type, thickness=thickness)
            x += pxstep

        while y < (self.height * scale_height - scale_height + 1):
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type, thickness=thickness)
            y += pystep
    
    
    def draw_agents(self, img, scale_length, scale_height, agent_color, agent_label_text_color):
        agent_counts = {}
        for agent in self.agents:
            x = agent.node.x * scale_length
            y = agent.node.y * scale_height
            if (x, y) not in agent_counts:
                agent_counts[(x, y)] = 1
            else:
                agent_counts[(x, y)] += 1

            shift_factor_x = (agent_counts[(x, y)] - 1) * int(scale_length / 10)
            shift_factor_y = (agent_counts[(x, y)] - 1) * int(scale_height / 10)
            pt1 = (x - int(scale_length / 5) + shift_factor_x, y + int(scale_height / 5) + shift_factor_y)
            pt2 = (x + int(scale_length / 5) + shift_factor_x, y + int(scale_height / 5) + shift_factor_y)
            pt3 = (x + shift_factor_x, y - int(scale_height / 5) + shift_factor_y)
            cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, agent_color, -1)
            cv2.putText(img, str(agent.label), (x + int(scale_height / 4) + shift_factor_x, y - int(scale_height / 4)+ shift_factor_y), cv2.FONT_HERSHEY_SIMPLEX, 1, agent_label_text_color, 2, cv2.LINE_AA)

    
    def render(self, image_size=800, grid_line_color=(87, 89, 93), grid_line_thickness=1, grid_line_type=cv2.LINE_AA, node_color=(255, 0, 0), terminating_node_color=(0, 0, 255), agent_color=(255, 255, 0), agent_label_text_color=(0, 255, 255)):
        scale_length = int(image_size / self.length)
        scale_height = int(image_size / self.height)

        img = np.zeros((self.length * scale_length, self.height * scale_height, 3), np.uint8)
        
        self.draw_grid(img, scale_length, scale_height, grid_line_color, grid_line_thickness, grid_line_type)
        numedges = 0
        for one, two, w in self.G.edges.data("weight"):
            if w != 0:
                if w != 1:
                    color = (255, 0, 255)
                else:
                    color = (0, 255, 0) 
                cv2.line(img, self.G.nodes[one]["val"].scale_coords(scale_length, scale_height), self.G.nodes[two]["val"].scale_coords(scale_length, scale_height), color = color, lineType=cv2.LINE_AA, thickness=1)
                numedges += 1
        print(numedges)

        for label in list(self.G.nodes.keys()):
            if (n := self.G.nodes[label]["val"]).terminating:
                cv2.circle(img, n.scale_coords(scale_length, scale_height), int(scale_length/3), terminating_node_color, -1)  
            else:
                cv2.circle(img, n.scale_coords(scale_length, scale_height), int(scale_length/3), node_color, -1)

        self.draw_agents(img, scale_length, scale_height, agent_color, agent_label_text_color)
        return img
        

adj = np.array([])
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video = cv2.VideoWriter('graphVideo.avi',fourcc, 2, (800,800))
print("started")
print("\nVideo Started")
done = False
nodes = random.randint(5, 32)
edges = random.randint(5, 32)
grid = GraphWorld(length=50, height=50, num_nodes=8, num_edges=16, fully_connected=True, n_agents=3, num_terminating_nodes=2, collaborative=True, ratio = 0.5, adv = 0.25)
#grid = GraphWorld(parse_args=True, adj=adj)
video.write(grid.render())
i = 0
while not done:
    observations, rewards, done = grid.step()
    print(observations, rewards)
    video.write(frame := grid.render())
    if i == 0:
        cv2.imwrite('gridTest.jpg', frame)
    i += 1
print("Video Finished!")

cv2.destroyAllWindows()
video.release()
