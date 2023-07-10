import gym
import numpy as np
import gym_examples
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import cv2

class GraphWorld():
    
    length = 0
    height = 0
    arr = np.array([])
    num_nodes = 0
    max_edges = 0
    adj = np.array([])
    
    def __init__(self, length, height, num_nodes, max_edges) -> None:
        self.length = length
        self.height = height
        self.num_nodes = num_nodes
        self.max_edges = max_edges
        self.adj = np.zeros(shape = (num_nodes, num_nodes))
        
        default_max_edges = 0
        for i in range(num_nodes - 1, 0, -1):
            default_max_edges += i
        
        if max_edges > default_max_edges:
            max_edges = default_max_edges  
            print(f"Max number of Edges inputted exceeds amount possible! Running with {max_edges} edges")

        num_edges = 0
        while len(self.arr) < self.num_nodes:
            node = Node(self.length, self.height, 1)
            b = True
            for n in self.arr:
                if n.return_coords() == node.return_coords():
                    b = False
            if b:
                self.arr = np.append(self.arr, node)

        visited = set()
        for i, node in enumerate(self.arr):
            print("I: ", i)
            ind = random.randint(0, num_nodes - 1)
            while ind == i:
                ind = random.randint(0, num_nodes - 1)
            print("Ind: ", ind)
            if ind not in visited and i not in visited:
                print("ran")
                self.undir_edge(node, self.arr[ind])
                num_edges += 1
                visited.add(ind)
                visited.add(i)
            print("visited: ", visited)
            print(" ")

        while num_edges < max_edges:
            ind1, ind2 = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if ind1 != ind2:
                if(self.undir_edge(self.arr[ind1], self.arr[ind2])):
                    num_edges += 1
        
        """
        for node in self.arr:
            print("NEW NODE:")
            print(node.return_coords())
            for i in node.get_neighbors():
                print("neigbor", end = '  ')
                print(i.return_coords())
        """

        for i, node in enumerate(self.arr):
            nbrs = node.get_neighbors()
            for n in nbrs:
                j = np.where(self.arr == n)[0][0]
                self.adj[i][j] = 1
        print(self.adj)
            #adj[coords[0]][coords[1]] = 1  
        self.render(800)
        
    def get_mat(self):
        return self.arr
    
    def undir_edge(self, node1, node2):
        return node1.add_edge(node2) and node2.add_edge(node1)       

    def get_val(self, y, x):
        return self.arr[y, x]
    
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
        
        visited = set()
        for node in self.arr:
            shape_idx = random.randint(0, 0)
            if shape_idx == 0:
                cv2.circle(img, node.scale_coords(scale_length, scale_height), int(scale_length/3), (255, 0, 0), -1)
            for nbr in node.get_neighbors():
                if nbr not in visited:
                    cv2.line(img, node.scale_coords(scale_length, scale_height), nbr.scale_coords(scale_length, scale_height), color=(0, 255, 0), lineType=cv2.LINE_AA, thickness=1)
                    visited.add(nbr)
            # elif shape_idx == 1:
            #     cv2.rectangle(img, node.scale_coords())
            # else:
            #     cv2.ellipse(img, node.scale_coords(scale_length, scale_height), scale_length/2, (255, 0, 0), -1) 
                

        cv2.imwrite('gridTest.jpg', img)
        

class Node(): 
    x_coord = 0
    y_coord = 0
    length = 0
    height = 0
    nbrs = np.array([])
    degree = 0
    
    def __init__(self, length, height, degree) -> None:
        self.x_coord = random.randint(0, length)
        self.y_coord = random.randint(0, height)
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
    
    def get_neighbors(self):
        return self.nbrs
        
    def add_edge(self, node):
        if node not in self.nbrs:
            self.nbrs = np.append(self.nbrs, node)
            return True
        return False
        """
        if node in self.nbrs and self in node.get_neighbors():
            return
        elif node not in self.nbrs and self not in node.get_neighbors():
            self.nbrs = np.append(self.nbrs, node)
            node.add_edge(self)
        elif self not in node.get_neighbors():
            test_nbrs = np.append(node.get_neighbors(), self)
            print("test nbrs: ", test_nbrs)
            node.set_neighbors(test_nbrs)
        """
        
grid = GraphWorld(20, 20, 7, 6)
#ratio, adv

     


# env = gym.make('gym_examples/GridWorld-v0', render_mode = "rgb_array")
# env.reset()
# plt.imshow(env.render())
# plt.show()
# time.sleep(2)
# for i in range(50):
#     action = random.randrange(0, 3)
#     observation, reward, terminated, truncated, info = env.step(action)
#     frame = env.render()    
#     clear_output(wait=True) 
#     plt.imshow(frame)
#     plt.show()
#     plt.clear()
#     print("ran")
#     time.sleep(0.5)
#     if terminated: break


# env = gym.make('gym_examples/GridWorld-v0')

# from gym.wrappers import FlattenObservation

# env = gym.make('gym_examples/GridWorld-v0')
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}

# from gym_examples.wrappers import RelativePosition

# env = gym.make('gym_examples/GridWorld-v0')
# wrapped_env = RelativePosition(env)
# print(wrapped_env.reset())     # E.g.  [-3  3], {}

# frame = env.render()
# print(type(frame))
# plt.imshow(frame)
# plt.show()
