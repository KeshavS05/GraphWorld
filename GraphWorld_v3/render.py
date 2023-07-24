import numpy as np
import cv2

class Render():
    
    def __init__(self, length, height, G, agents, weight_before, weight_after) -> None:
        self.length = length
        self.height = height
        self.scale_length = 0
        self.scale_height = 0
        self.G = G
        self.nodes = self.G.nodes
        self.edges = self.G.edges(data="weight")
        self.agents = agents
        self.weight_before = weight_before
        self.weight_after = weight_after
        

    def get_node(self, node_label):
        return self.nodes[node_label]["val"]

    def render_grid(self, img, line_color, thickness, type):
        x = self.scale_length
        y = self.scale_height
        while x < (self.length * self.scale_length - self.scale_length + 1):
            cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type, thickness=thickness)
            x += self.scale_length

        while y < (self.height * self.scale_height - self.scale_height + 1):
            cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type, thickness=thickness)
            y += self.scale_height

    def render_edges(self, img, default_edge_color, active_support_edge_color, inactive_support_edge_color,):
        for x, y, w in self.edges:
            if w != 0:
                if w == self.weight_before:
                    cv2.line(img, self.get_node(x).scale_coords(self.scale_length, self.scale_height), self.get_node(y).scale_coords(self.scale_length, self.scale_height), color=inactive_support_edge_color, lineType=cv2.LINE_AA, thickness=1)
                elif w == 1:
                    cv2.line(img, self.get_node(x).scale_coords(self.scale_length, self.scale_height), self.get_node(y).scale_coords(self.scale_length, self.scale_height), color=default_edge_color, lineType=cv2.LINE_AA, thickness=1)
                else:
                    cv2.line(img, self.get_node(x).scale_coords(self.scale_length, self.scale_height), self.get_node(y).scale_coords(self.scale_length, self.scale_height), color=active_support_edge_color, lineType=cv2.LINE_AA, thickness=1)

    def render_nodes(self, img, image_size, node_color, node_border_color, terminating_node_color, supporting_node_color):
        for label in list(self.G.nodes.keys()):
            n = self.get_node(label)
            cv2.circle(img, n.scale_coords(self.scale_length, self.scale_height), int(image_size/(len(self.nodes) * 2)), node_border_color, 5)
            if n.terminating:
                cv2.circle(img, n.scale_coords(self.scale_length, self.scale_height), int(image_size/(len(self.nodes) * 2)), terminating_node_color, -1)
            elif n.supporting:
                cv2.circle(img, n.scale_coords(self.scale_length, self.scale_height), int(image_size/(len(self.nodes) * 2)), supporting_node_color, -1)
            else:
                cv2.circle(img, n.scale_coords(self.scale_length, self.scale_height), int(image_size/(len(self.nodes) * 2)), node_color, -1)

    def render_agents(self, img, image_size, agent_color, agent_label_text_color):
        agent_counts = {}
        for agent in self.agents:
            x = agent.node.x * self.scale_length
            y = agent.node.y * self.scale_height
            if (x, y) not in agent_counts:
                agent_counts[(x, y)] = 1
            else:
                agent_counts[(x, y)] += 1

            shift_factor_x = (agent_counts[(x, y)] - 1) * int(image_size/(len(self.nodes) * 6))
            shift_factor_y = (agent_counts[(x, y)] - 1) * int(image_size/(len(self.nodes) * 6))
            pt1 = (x - int(image_size/(len(self.nodes) * 3)) + shift_factor_x, y + int(image_size/(len(self.nodes) * 3)) + shift_factor_y)
            pt2 = (x + int(image_size/(len(self.nodes) * 3)) + shift_factor_x, y + int(image_size/(len(self.nodes) * 3)) + shift_factor_y)
            pt3 = (x + shift_factor_x, y - int(image_size/(len(self.nodes) * 3)) + shift_factor_y)
            cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, agent_color, -1)
            cv2.putText(img, str(agent.label), (x + shift_factor_x, y + shift_factor_y), cv2.FONT_HERSHEY_SIMPLEX, 1, agent_label_text_color, 2, cv2.LINE_AA)


    def render(self, image_size, grid_line_color, grid_line_thickness, grid_line_type, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, node_border_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color):
        self.scale_length = int(image_size / self.length)
        self.scale_height = int(image_size / self.height)
        img = np.zeros((self.length * self.scale_length, self.height * self.scale_height, 3), np.uint8)  
        self.render_grid(img, grid_line_color, grid_line_thickness, grid_line_type)
        self.render_edges(img, default_edge_color, active_support_edge_color, inactive_support_edge_color)
        self.render_nodes(img, image_size, node_color, node_border_color, terminating_node_color, supporting_node_color)  
        self.render_agents(img, image_size, agent_color, agent_label_text_color)
        return img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
