import numpy as np
import random
import cv2
from GraphWorld import GraphWorld
import keyboard
from threeD_render import threeD_Render
import time

adj = np.array([])
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video = cv2.VideoWriter('renders/graphVideo.avi',fourcc, 1, (800,800))
print("started")
print("\nVideo Started")
done = False
nodes = random.randint(5, 32)
edges = random.randint(5, 32)
grid = GraphWorld(threeD=True, length=10, height=10, depth=10, num_nodes=200, num_edges=400, fully_connected=True, n_agents=8, num_terminating_nodes=2, collaborative=True, ratio = 0.25, weight_before = 1.5, weight_after = 0.5)
grid.reset()
grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
#r = threeD_Render(grid.length, grid.height, grid.depth, grid.G, grid.agents, grid.weight_before, grid.weight_after)
#r.render(grid_line_color=(0, 0, 255), grid_line_thickness=10, grid_line_type=cv2.LINE_AA, grid_tick_val=1, default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 255, 255))
#grid = GraphWorld(parse_args=True, adj=adj)
# video.write(grid.render())
# i = 0
# frames = np.array([grid.render()])
# max_i = 0
# while True:
#     print("true")
#     if keyboard.wait('d'):
#         print('hi')
#         i += 1
#         if i > max_i:
#             max_i = i
#             observations, rewards, done = grid.step() 
#             #frame = r.render() 
#             r = threeD_Render(grid.length, grid.height, grid.scale_length, grid.scale_height, grid.G, grid.agents, grid.weight_before, grid.weight_after)
#             frame = r.render(image_size=1500, grid_line_color=(0, 0, 255), grid_line_thickness=100, grid_line_type=cv2.LINE_AA, default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 255, 255))
#             frames.append(frame)
#         else:
#             frame = frames[i]
#         cv2.imwrite('renders/gridTest.jpg', frame)
#     elif keyboard.wait('a') or keyboard.wait('left arrow'):
#         if i == 0:
#             continue
#         i -= 1
#         frame = frames[i] 
    
#         cv2.imwrite('renders/gridTest.jpg', frame)
#     elif keyboard.wait('x'):
#         break             
"""
    observations, rewards, done = grid.step()
    video.write(frame := grid.render())
    if i == 0:
        cv2.imwrite('renders/gridTest.jpg', frame)
    i += 1
    if :
        user_active = True
"""
print("Video Finished!")
#print(i)

cv2.destroyAllWindows()
video.release()