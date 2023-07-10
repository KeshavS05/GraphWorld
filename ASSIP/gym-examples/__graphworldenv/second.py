import gym
import networkx as nx
import numpy as np
import gym_examples
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import cv2

class GraphWorld():
    
    length = 0
    height = 0
    G = nx.Graph()
    num_nodes = 0
    adj = np.array([])
    random = False
    edges = 0
    num_nodes = 0
    
    def __init__(self, length, height, adj, random, num_nodes, edges) -> None:
        self.length = length
        self.height = height
        self.random = random
        self.edges = edges
        self.num_nodes = num_nodes
        if self.random == True:
            self.adj = self.random_adj(self.num_nodes, self.edges)
        else:
            self.adj = adj
            self.num_nodes = len(self.adj)
        node_label = 0
        coords_occupied = []
        while self.G.number_of_nodes() < self.num_nodes:
            node = Node(self.length, self.height, 1)
            if node.return_coords() not in coords_occupied:
                self.G.add_node(node_label, val = node, coord = node.return_coords())
                node_label += 1
                coords_occupied.append(node.return_coords())

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                weight = self.adj[i][j]
                if weight > 0:
                    self.G.add_weighted_edges_from([(i, j, weight)])

        self.render(800)
        
        
    def random_adj(self, num, edges):
        adjacent = np.zeros((num, num))
        possible = len(np.triu_indices(num, 1)[0])
        if edges > possible:
            print("The amount of edges exceeds those possible!")
            print(f"Edges reset to: {possible}")
            edges = possible
        indicies = np.triu_indices(num, 1)
        visited = set()
        for i in range(edges):
            idx = random.randint(0, possible - 1)
            while idx in visited:
                idx = random.randint(0, possible - 1)
            visited.add(idx)
            adjacent[indicies[0][idx]][indicies[1][idx]] = 1
        adjacent = adjacent + adjacent.T
        return adjacent
        
            
            

    def draw_grid(self, img, scale_length, scale_height, line_color=(87, 89, 93), thickness=1, type_=cv2.LINE_AA):
        pxstep = scale_length
        pystep = scale_height
        x = scale_length
        y = scale_height
        while x < img.shape[1]:
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
            x += pxstep

        while y < img.shape[0]:
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
            y += pystep
    
    def render(self, image_size):
        if image_size / self.length == int(image_size/self.length):
            scale_length = int(image_size / self.length)
        else:
            scale_length = int(image_size / self.length) + 1

        if image_size / self.height == int(image_size/self.height):
            scale_height = int(image_size / self.height)
        else:
            scale_height = int(image_size / self.height) + 1   


        img = np.zeros((self.length * scale_length, self.height * scale_height, 3), np.uint8)
        
        self.draw_grid(img, scale_length, scale_height)
        
        for label in list(self.G.nodes.keys()):
            shape_idx = random.randint(0, 0)
            for nbr in list(self.G.neighbors(label)):
                cv2.line(img, self.G.nodes[label]["val"].scale_coords(scale_length, scale_height), self.G.nodes[nbr]["val"].scale_coords(scale_length, scale_height), color=(0, 255, 0), lineType=cv2.LINE_AA, thickness=1)
            if shape_idx == 0:
                cv2.circle(img, self.G.nodes[label]["val"].scale_coords(scale_length, scale_height), len(list(self.G.neighbors(label))) * 3, (255, 0, 0), -1)
                cv2.putText(img, str(label), self.G.nodes[label]["val"].scale_coords(scale_length, scale_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
#int(scale_length/3)
        cv2.imwrite('gridTest.jpg', img)
        print(nx.to_numpy_array(self.G))
        

class Node(): 
    x_coord = 0
    y_coord = 0
    length = 0
    height = 0
    nbrs = np.array([])
    degree = 0
    
    def __init__(self, length, height, degree) -> None:
        self.x_coord = random.randint(1, length - 1)
        self.y_coord = random.randint(1, height - 1)
        self.degree = degree
    
    def return_coords(self):
        return (self.x_coord, self.y_coord)
    
    def scale_coords(self, scale_length, scale_height):
        tup = self.return_coords()
        x = tup[0] * scale_length
        y = tup[1] * scale_height
        return (x, y)
    
    def return_deg(self):
        return self.degree
        
    def equals(self, node):
        return self.return_coords == node.return_coords

adj = []
grid = GraphWorld(50, 50, adj, False, num_nodes = 7, edges = 50)

