import numpy as np

class Node(): 

    def __init__(self, node_label, length, height, depth, terminating, supporting, num_agents=0) -> None:
        self.node_label = node_label
        self.x = np.random.randint(1, length)
        self.y = np.random.randint(1, height)
        self.z = np.random.randint(1, depth)
        self.terminating = terminating
        self.supporting = supporting
        self.supporting_edges = []
        self.num_agents = num_agents

    
    def return_coords(self):
        return (self.x, self.y, self.z)
    
    def scale_coords(self, scale_length, scale_height):
        tup = self.return_coords()
        x = tup[0] * scale_length
        y = tup[1] * scale_height
        return (x, y)
        
    def equals(self, node):
        return self.return_coords == node.return_coords