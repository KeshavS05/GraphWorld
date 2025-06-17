import numpy as np
from GraphWorldCopyCopy import GraphWorld
import time
import dash
from dash import ctx
from dash.dependencies import Output, Input, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import os
import webbrowser
from threading import Timer
import sys; args = sys.argv[1:]
import itertools
import pickle
from itertools import product
from our_jsg import JSGraph
from graphCompare import shortest_path_cost

threeD = True

# five_node_edges = {(0, 1):1, (0, 2):0, (1, 2):0,(1, 3):0, (2, 3):1, (3, 4):0} ## dense graph 
# five_node_risk_edges_with_support_nodes = {(0, 1):(0, 1), (2, 3): (2, 3)}         
        

# six_node_edges = {(0, 1): 0, (0, 2): 0, (0, 3): 1, (1, 3): 0, (2, 3): 0, (3, 4): 1, (4, 5): 0}
# six_node_risk_edges_with_support_nodes = {(0, 3): (1, 2), (3, 4): (3, 4)}
    

# ten_node_edges = {(0, 1): 0, (0, 2): 0, (0, 3): 1, (1, 2): 0,(1, 4): 0,(1, 5): 0, (2, 3): 0,(2, 4): 0, 
#                     (2, 6): 0, (3, 6): 0, (3, 5): 0, (3, 4): 0, (4, 5): 0, (4, 6): 0, 
#                     (5, 7): 1, (5, 6): 0, (6, 7): 0, (6, 8): 0, (7, 8): 0, (7, 9): 0,
#                     (8, 4): 0, (8, 9): 0}
# ten_node_risk_edges_with_support_nodes = {(0, 3):(0, 3), (5, 7): (5, 7)} # write a code to get it from edges

# twenty_node_edges = {(0, 1): 0, (0, 2): 0, (0, 3): 1, (1, 3): 0, (2, 3): 0, (3, 4): 1, (4, 5): 0,
#         (1, 6): 0, (2, 8): 1, (3, 6): 0, (3, 8): 0, (4, 6): 1, (6, 7): 0, (4, 7): 0,  (4, 9): 0,
#         (8, 4): 0, (8, 9): 0,(5, 7): 0, (5, 9): 0,
#         (1, 10): 0, (6, 10): 0, (6, 11): 0, (10, 11): 0, (7, 11): 0, (7, 12): 0, (11, 12): 0, (6, 17): 0,
#         (12, 13): 0, (12, 14): 0, (12, 15): 0, (13, 15): 0, (5, 17): 0, (14, 18): 0, (15, 19): 0, (16, 19): 0,(18, 19): 0,(16, 17): 0,
#         (7, 13): 1, (5, 16): 1, (14, 15): 1}

# twenty_node_risk_edges_with_support_nodes = {(0, 3): (2, 2), (3, 4): (8, 8),(2, 8): (3, 3), (4, 6): (3, 3),
#                                  (7, 13): (16, 16), (5, 16): (7, 7), (14, 15): (15, 15)} # write a code to get it from edges

# thirty_node_edges = {(0, 1): 0, (0, 2): 0, (0, 3): 1, (1, 3): 0, (2, 3): 0, (3, 4): 1, (4, 5): 0,
#         (1, 6): 0, (2, 8): 1, (3, 6): 0, (3, 8): 0, (4, 6): 1, (6, 7): 0, (4, 7): 0,  (4, 9): 0,
#         (8, 4): 0, (8, 9): 0,(5, 7): 0, (5, 9): 0,
#         (1, 10): 0, (6, 10): 0, (6, 11): 0, (10, 11): 0, (7, 11): 0, (7, 12): 0, (11, 12): 0, (6, 17): 0,
#         (12, 13): 0, (12, 14): 0, (12, 15): 0, (13, 15): 0, (5, 17): 0, (14, 18): 0, (15, 19): 0, (16, 19): 0,(18, 19): 0,(16, 17): 0,
#         (7, 13): 1, (5, 16): 1, (14, 15): 1,  (20, 21): 0, (20, 22): 0, (9, 22): 0, (22, 23): 0,
#         (23, 8): 0, (23, 24): 1, (24, 2): 0, (24, 25): 0, (25, 26): 0, (26, 27): 0, (26, 23): 0,
#         (27, 28): 0, (21, 29): 0, (29, 28): 0,(22, 29): 0, (17, 20): 1, (23, 28): 1, (8, 22): 1}

# thirty_node_risk_edges_with_support_nodes = {(0, 3): (2, 2), (3, 4): (8, 8), (2, 8): (3, 3), (4, 6): (3, 3),
#                                  (7, 13): (16, 16), (5, 16): (7, 7), (14, 15): (15, 15),
#                                  (23, 24): (19, 19), (23, 28): (22, 22), (8, 22): (22, 22), (9, 22): (23, 23),(17, 20): (17, 17)} # write a code to get it from edges
#                                 #  (22, 9): (20, 20), (17, 20): (16, 16), (23, 24): (21, 21)



all_graphs = {
    '5_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        'edges': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1}, #path: 0, 1, 2, 3, 4
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (2, 3): (2, 3)}, # risk_edges = at least 1/2, 1/3, 1/5 of total edges
        'type_of_nodes': "sparse"
    },
     '5_node_moderate': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        'edges': {(0, 1):1, (0, 2):1, (1, 2):1, (2, 3):1, (3, 4):1}, #path: 0, 2, 3, 4
        'risk_edges_with_support_nodes': {(0, 2): (0, 2), (2, 3): (2, 3)},
        'type_of_nodes': "moderate"
    },
    '5_node_dense': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        'edges': {(0, 3):1, (1, 2):1, (3, 1):1, (1, 4):1, (3, 2):1, (2, 4):1}, #path: 0, 3, 4 (3 steps to goal)
        'risk_edges_with_support_nodes': {(0, 3): (0, 3), (3, 1): (3, 1), (1, 4): (1, 4), (3, 2): (3, 2)},
        'type_of_nodes': "dense"
    },



    '10_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9},
        'edges': {(0, 1):1, (1, 2):1, (2, 3):1, (3, 4):1, (4, 5):1, (5, 6):1, (6, 7):1, (7, 8):1, (8, 9):1}, #path: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ( 10 steps)
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (2, 3): (2, 3),(4, 5): (4, 5), (6, 7):(6, 7)},
        'type_of_nodes': "sparse"
    },
     '10_node_moderate': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 4): 1, (1, 5): 1,
                   (2, 3): 1, (2, 5): 1, (2, 6): 1, (3, 6): 1, (4, 5): 1, (4, 8): 1,
                   (5, 6): 1, (5, 7): 1, (5, 8): 1, (6, 7): 1, (7, 8): 1, (8, 9): 1}, # 5 nodes to goal
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (0, 2): (0, 2), (3, 6): (3, 6), (5, 8): (5, 8)},
        'type_of_nodes': "moderate"
    },
    '10_node_dense': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1,(1, 4): 1,(1, 5): 1, 
                  (2, 3): 1,(2, 4): 1, (2, 5):1, (2, 6): 1, (3, 7): 1, (3, 6): 1, (3, 5): 1, 
                  (3, 4): 1, (4, 5): 1, (4, 6): 1,(4, 7): 1, (5, 7): 1, (5, 6): 1, (6, 7): 1, 
                  (6, 8): 1, (7, 8): 1, (7, 9): 1, (8, 4): 1, (8, 9): 1}, # path: 5 nodes to goal
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (2, 4): (2, 4), (3, 4): (3, 4), (3, 5): (3, 5), (3, 7): (3, 7)},
        'type_of_nodes': "sparse"
    },


    '15_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14},
        'edges': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
                        (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (3, 4): (3, 4), (5, 6): (5, 6), (9, 10): (9, 10), (12, 13): (12, 13)},
        'type_of_nodes': "sparse"
    },

    '15_node_moderate': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14},
        'edges': {(0, 1): 1, (0, 2): 1,(0, 3): 1, (1, 2): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
                (2, 4): 1, (3, 6): 1, (3, 8): 1, (4, 7): 1, (4, 5): 1, (4, 8): 1, (4, 9): 1, 
                (5, 7): 1, (5, 8): 1, (6, 7): 1, (6, 8): 1,(6, 13): 1,  (7, 8): 1, (7, 9): 1,
                (8, 4): 1, (8, 12):1, (8, 13):1, (9, 10):1, (9, 13):1, (10, 12):1, (10, 14):1, (11, 12):1,
                (11, 13):1, (11, 14):1, (12, 13):1,(12, 14):1}, # path (6-7 steps to goal)
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (0, 3): (0, 3), (4, 7): (4, 7),(6, 13): (6, 13), (8, 12): (8, 12)},
        'type_of_nodes': "moderate"
    },  
    '15_node_dense': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14},
        'edges':  {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 4):1,(1, 5):1,(1, 6):1, (1, 2):1,
                   (2, 3):1, (2, 4):1, (2, 5):1, (2, 6):1, (3, 6):1, (3, 4):1, 
                   (4, 5):1, (4, 8):1, (4, 10):1, (5, 6):1,(5, 7):1, (5, 8):1, (5, 9):1,
                   (6, 9):1, (6, 10):1, (6, 7):1,(7, 10):1, (7, 12):1, (7, 13):1,(8, 9):1, 
                   (8, 10):1, (8, 11):1, (8, 12):1,(9, 12):1, (9, 11):1, (9, 10):1, 
                   (10, 12):1, (11, 12):1,(11, 13):1,(12, 13):1,(13, 14):1,}, # path (6-7 steps to goal)
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (2, 6): (2, 6), (4, 8): (4, 8), (5, 9): (5, 9), (10, 12): (10, 12),(11, 13): (11, 13)},
        'type_of_nodes': "dense",
    },

    '20_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19},
        'edges': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
                (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1, (14, 15): 1, (15, 16): 1,
                (16, 17): 1, (17, 18): 1, (18, 19): 1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
                                                (16, 17): (16, 17)},
        'type_of_nodes': "sparse",
    },
    '20_node_moderate': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 4): 1, (0, 6): 1,(1, 2): 1,(1, 3): 1, (2, 3): 1,(2, 4): 1, 
        (3, 6): 1, (3, 8): 1, (3, 4): 1, (3, 7): 1,(4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1, 
        (5, 7): 1, (5, 10): 1, (6, 7): 1,(6, 9): 1, (7, 9): 1,(7, 10): 1,(7, 11): 1,(8, 7): 1, 
        (8, 10):1, (8, 11):1, (8, 13):1, (8, 14):1, (8, 15):1, (8, 16):1,(9, 10):1, (9, 11):1, 
        (9, 13):1, (10, 12):1, (10, 14):1,(11, 14):1,(11, 15):1,(11, 13):1, (12, 13):1,(12, 16):1,
        (13, 14):1, (13, 15):1, (13, 18):1,(14, 15):1, (14, 16):1, (14, 17):1,(14, 19):1, (15, 17):1, 
        (15, 18):1,(16, 19):1,  (17, 18):1, (17, 19):1, (18, 19):1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (0, 4): (0, 4),  (7, 8): (7, 8), (9, 10): (9, 10), (12, 16): (12, 16),
                                                (14, 17): (14, 17)},
        'type_of_nodes': "moderate",
    },

    '20_node_dense': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 4): 1, (0, 6): 1,(1, 2): 1,(1, 3): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
                (2, 10): 1, (3, 6): 1, (3, 8): 1,(3, 12): 1, (3, 4): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1, 
                (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 8): 1, (6, 9): 1, (7, 8): 1, (7, 9): 1,(7, 10): 
                1,(7, 11): 1,(7, 13): 1,(8, 7): 1, (8, 9): 1, (8, 10):1, (8, 11):1, (8, 12):1,(8, 13):1, (8, 14):1, 
                (8, 15):1, (8, 16):1,(8, 17):1,(8, 18):1,(8, 19):1,(9, 10):1, (9, 11):1, (9, 13):1, (9, 14):1, 
                (10, 12):1, (10, 14):1, (11, 12):1, (11, 14):1,(11, 15):1, (11, 16):1, (11, 19):1,
                (11, 13):1, (12, 13):1,(12, 14):1, (12, 16):1, (12, 18):1, (13, 14):1, (13, 15):1, (13, 18):1,
                (14, 15):1, (14, 16):1, (14, 17):1, (14, 18):1, (14, 19):1, (15, 16):1, (15, 17):1, (15, 18):1,
                (16, 17):1,  (17, 18):1, (17, 19):1, (18, 19):1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
                                              (12, 18): (12, 18), (16, 17): (16, 17)},
        'type_of_nodes': "dense",
    },

    '25_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14,15:15, 16:16, 17:17, 
                  18:18, 19:19,20:20, 21:21, 22:22, 23:23, 24:24},
        'edges': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
                (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1, (14, 15): 1, (15, 16): 1,
                (16, 17): 1, (17, 18): 1, (18, 19): 1, (19, 20):1, (20, 21):1, (21, 22):1, (22, 23):1, (23, 24):1 },
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
                                        (15, 16): (15, 16), (18, 19): (18, 19), (21, 22): (21, 22)},
        'type_of_nodes': "sparse",
    },
    '25_node_moderate': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14,15:15, 16:16, 17:17,
                    18:18, 19:19,20:20, 21:21, 22:22, 23:23, 24:24},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 5): 1,(0, 6): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1, (2, 4): 1, 
                (2, 8): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1,(3, 10): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1,
                (4, 11): 1,(5, 6): 1, (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 9): 1, 
                (7, 8): 1, (7, 9): 1,(7, 10): 1,(7, 11): 1,(7, 13): 1,(8, 9): 1, 
                (8, 12):1,(8, 13):1, (8, 14):1, (8, 15):1, (8, 16):1,(8, 17):1,
                (9, 11):1, (9, 14):1, (10, 12):1, (10, 14):1,(10, 16):1,(11, 14):1,
                (11, 15):1, (11, 13):1, (12, 14):1, (12, 16):1, (12, 18):1,
                (13, 15):1, (13, 18):1, (13, 21):1, (14, 17):1, (14, 18):1, (14, 19):1,(14, 21):1,
                (15, 17):1, (15, 18):1, (17, 19):1, (17, 23):1,(18, 19):1, 
                (18, 20):1, (19, 22):1,(18, 21):1, (18, 22):1, 
                (20, 21):1, (19, 23):1, (19, 24):1, (20, 22):1, (20, 23):1, (21, 22): 1, (22, 24):1,(23, 24):1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (0, 2): (0, 2), (0, 6): (0, 6),(3, 4): (3, 4), (6, 7): (6, 7),
                                          (9, 11): (9, 11), (10, 12): (10, 12),(12, 14): (12, 14),(17, 19): (17, 19),
                                          (15, 18): (15, 18), (21, 20): (21, 20), (21, 22): (21, 22)},
        'type_of_nodes': "moderate",
    },

    '25_node_dense': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14,15:15, 16:16, 17:17,
                    18:18, 19:19,20:20, 21:21, 22:22, 23:23, 24:24},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 5): 1,(0, 6): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
                (2, 8): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1,(3, 10): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1,
                (4, 11): 1, (4, 13): 1, (5, 6): 1, (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 8): 1, (6, 9): 1, 
                (7, 8): 1, (7, 9): 1,(7, 10): 1,(7, 11): 1,(7, 13): 1,(8, 4): 1, (8, 9): 1, (8, 10):1,(8, 11):1, 
                (8, 12):1,(8, 13):1, (8, 14):1, (8, 15):1, (8, 16):1,(8, 17):1,(8, 18):1,(8, 19):1, (8, 19): 1,
                (9, 10):1, (9, 11):1, (9, 13):1, (9, 14):1, (10, 12):1, (10, 14):1,(10, 16):1, (11, 12):1, (11, 14):1,
                (11, 15):1, (11, 16):1, (11, 19):1,(11, 13):1, (12, 13):1,(12, 14):1, (12, 16):1, (12, 18):1, (13, 14):1,
                (13, 15):1, (13, 18):1, (13, 21):1, (14, 15):1, (14, 16):1, (14, 17):1, (14, 18):1, (14, 19):1,(14, 21):1,
                (15, 16):1, (15, 17):1, (15, 18):1, (13, 22):1,(16, 17):1,  (17, 18):1, (17, 19):1, (17, 23):1,(18, 19):1, 
                (18, 19):1, (18, 20):1, (18, 19):1, (18, 20):1, (19, 20):1, (19, 22):1,(18, 21):1, (18, 22):1, 
                (18, 23):1, (18, 24):1, (20, 21):1, (19, 23):1, (19, 24):1, (20, 22):1, (20, 23):1, (20, 24):1, (23, 24):1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (0, 3): (0, 3), (0, 5): (0, 5), (6, 7): (6, 7), (7, 8): (7, 8),
                                          (8, 13):(8, 13), (8, 16):(8, 16),(9, 10): (9, 10), (10, 12):  (10, 12),(12, 13): (12, 13),
                                          (13, 14):(13, 14),(11, 19):(11, 19), (10, 16): (10, 16),(14, 21): (14, 21), (17, 18): (17, 18),
                                          (18, 20): (18, 20),(18, 23): (18, 23),(19, 22):(19, 22), (20, 23): (20, 23) },
        'type_of_nodes': "dense",
    },
      

    '30_node_sparse': {
        'nodes': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14,15:15, 16:16, 17:17,
                    18:18, 19:19,20:20, 21:21, 22:22, 23:23, 24:24, 25:25, 26:26, 27:27, 28:28, 29:29},
        'edges': {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
                (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1, (14, 15): 1, (15, 16): 1,
                (16, 17): 1, (17, 18): 1, (18, 19): 1, (19, 20):1, (20, 21):1, (21, 22):1, (22, 23):1, (23, 24):1,
                (24, 25):1, (25, 26):1, (26, 27):1, (27, 28):1, (28, 29):1},
        'risk_edges_with_support_nodes': {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
                                          (15, 16): (15, 16), (18, 19): (18, 19), (21, 22): (21, 22), (24, 25): (24, 25), (27, 28): (27, 28)},
        'type_of_nodes': "sparse",
    },

    '30_node_moderate': {
        'nodes':  {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,5:5, 6:6, 7:7, 8:8, 9:9,10:10, 11:11, 12:12, 13:13, 14:14,15:15, 16:16, 17:17,
                    18:18, 19:19,20:20, 21:21, 22:22, 23:23, 24:24, 25:25, 26:26, 27:27, 28:28, 29:29},
        'edges': {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 3): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1,
                (1, 6): 1, (2, 8): 1, (3, 6): 1, (3, 8): 1, (4, 6): 1, (6, 7): 1, (4, 7): 1,  (4, 9): 1,
                (8, 4): 1, (8, 9): 1,(5, 7): 1, (5, 9): 1,
                (1, 10): 1, (6, 10): 1, (6, 11): 1, (10, 11): 1, (7, 11): 1, (7, 12): 1, (11, 12): 1, (6, 17): 1,
                (12, 13): 1, (12, 14): 1, (12, 15): 1, (13, 15): 1, (13, 19): 1,(5, 17): 1, (14, 18): 1, (15, 19): 1, (16, 19): 1,(18, 19): 1,(16, 17): 1,
                (7, 13): 1, (5, 16): 1, (14, 15): 1, (20, 22): 1,(20, 29): 1, (22, 9): 1, (22, 23): 1,
                (23, 8): 1, (23, 24): 1, (24, 2): 1, (24, 25): 1, (25, 26): 1, (26, 27): 1, (26, 23): 1,
                (27, 28): 1,  (27, 21): 1,(29, 21): 1, (29, 28): 1,(22, 28): 1, (17, 20): 1, (23, 28): 1, (22, 8): 1},
        'risk_edges_with_support_nodes': {(0, 3): (0, 3), (3, 4): (3, 4), (2, 8): (2, 8), (4, 6): (4, 6), (7, 13): (7, 13),
                                          (5, 16):(5, 16), (14, 15): (14, 15), (23, 24): (23, 24), (23, 28):(23, 28), (22, 8):(22, 8),
                                          (22, 9):(22, 9), (17, 20): (17, 20)},
        'type_of_nodes': "moderate",
    }

    

}

# jsg = JSGraph(5)
# jsg.source = (0, 0)
# jsg.destination = (4, 4)
# jsg.graph_1d = adj_ws

# jsg_gc_start = time.time()
# jsg.trans_Env_To_JSG()
# print(jsg.nodesets)
# jsg_gc_end = time.time()

# jsg_spc_start = time.time()
# jsg_spc= shortest_path_cost(V, adjList_2D=jsg.get_adjMatrix_2D_JSG(), iter=V*V, S11=S11, Sgg=Sgg)
# jsg_spc_end = time.time()




# ########### Time Comparision JSG and CJSG ##########
# print("----------------JSG Outputs--------------")
# print(jsg.graph)
# print("Graph Construction Time: {}".format(jsg_gc_end-jsg_gc_start))
# print("Shortest Path Cost: {}".format(jsg_spc))
# print("Shortest Path Time (ms): {}".format((jsg_spc_end - jsg_spc_start)))
# print("Total Time Taken: {}".format((jsg_spc_end - jsg_spc_start)+(jsg_gc_end-jsg_gc_start)))
# print("----------------------------------------")

def plot_graph_test(episodes, rewards, num_nodes, n_agents, type_of_graph, type):
    import matplotlib.pyplot as plt
    if type == 'train':
        #file_path = "renders/large_multi_agents_riksy_graphs_train.png"
        file_path = "renders/" + type_of_graph + str(num_nodes) + "nodes" + str(n_agents) + "agents_reward_vs_episode_train.png"
        plt.title("train_" + type_of_graph + "_" + str(num_nodes) + "_nodes_" + str(n_agents) + "_agents")
    elif type == 'test':
        #file_path = "renders/large_multi_agents_risky_graphs_test.png"
        file_path = "renders/" + type_of_graph + str(num_nodes) + "nodes" + str(n_agents) + "agents_reward_vs_episode_test.png"
        plt.title("test_" + type_of_graph + "_" + str(num_nodes) + "_nodes_" + str(n_agents) + "_agents")
    else:
        raise ValueError('Invalid type')
        
    plt.plot(episodes, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(file_path) 
    plt.clf()
    # import matplotlib.pyplot as plt

    # file_path = "renders/" + str(type) + "/" + str(num_nodes) + "nodes" + str(n_agents) + "agents_reward_vs_episode_train_random.png"
    # plt.title(type+"_" + str(num_nodes) + "_nodes_" + str(n_agents) + "_agents_random")
    # plt.plot(episodes, rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.grid(True)
    # plt.savefig(file_path) 
# [[0, [10, ()], inf, [10, ()], inf],
#  [[10, ()], 0, [10, ()], inf, inf], 
#  [inf, [10, ()], 0, inf, inf], 
#  [[10, ()], inf, inf, 0, inf], 
#  [inf, inf, inf, inf, 0]]
#adj_ws = [[0, 0, 0, [10, (0, 3)], 0], [0, 0, [10, ()], [10, (3, 1)], [10, (1, 4)]], [0, [10, ()], 0, [10, (3, 2)], [10, ()]], [[10, (0, 3)], [10, (3, 1)], [10, (3, 2)], 0, 0], [0, [10, (1, 4)], [10, ()], 0, 0]]
num_nodes = 30
num_agents = 2
train_episodes = 1000
end_goals = np.array([num_nodes - 1]*num_agents)
type_of_graph = 'dense'
grid = GraphWorld(JSG=True, weight_before=10, weight_after=10, invalid_edge_weight=0, default_weight=10, threeD=True, html=True, length=100, height=100, depth=100, num_nodes=num_nodes, num_edges=8, fully_connected=True, n_agents=num_agents, num_terminating_nodes=1, collaborative=True, ratio=0.5, start_goals=np.array([0, 0, 0, 0, 0]), end_goals=end_goals)
grid.input_adj(all_graphs[str(num_nodes)+'_node_'+str(type_of_graph)]['edges'], all_graphs[str(num_nodes)+'_node_'+str(type_of_graph)]['risk_edges_with_support_nodes'])
#print(grid.adj)


jsg = JSGraph(num_nodes, grid.weight_after)
jsg.source = tuple([0]*num_agents)
jsg.destination = tuple([num_nodes - 1]*num_agents)
jsg.graph_1d = grid.adj

jsg_gc_start = time.time()
jsg.trans_Env_To_JSG(n_agents=num_agents)
# jsg.trans_Env_To_JSG()
jsg_gc_end = time.time()

jsg_spc_start = time.time()
it = num_nodes**num_agents
jsg_spc= shortest_path_cost(num_nodes, graph=jsg.get_adjMatrix_2D_JSG(), iter=it, S11=jsg.source, Sgg=jsg.destination)
jsg_spc_end = time.time()




########### Time Comparision JSG and CJSG ##########
print("----------------JSG Outputs--------------")
print(jsg.graph)
print("Graph Construction Time: {}".format(jsg_gc_end-jsg_gc_start))
print("Shortest Path Cost: {}".format(jsg_spc))
print("Shortest Path Time (ms): {}".format((jsg_spc_end - jsg_spc_start)))
print("Total Time Taken: {}".format((jsg_spc_end - jsg_spc_start)+(jsg_gc_end-jsg_gc_start)))
print("----------------------------------------")
time.sleep(10)
state, test_episode_reward, test_episode_steps, test_episdoe_actions = grid.reset(),0,0, []


def save_q_value(Q_values):
    file_name = "q_values_risk_large.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(Q_values, f)

    print("Q-values saved to", file_name)

start_train = time.time()
n_actions = grid.n_agents
num_nodes = grid.num_nodes
num_states = num_nodes ** grid.n_agents
nodes = grid.G.nodes

states_pairs = list(itertools.product(nodes, repeat=n_actions))
states_value_index = {state: index for index, state in enumerate(states_pairs)}

Q = np.zeros((num_states, num_states))

alpha = 0.8
gamma = 0.95
# Define hyperparameters
epsilon_initial = 0.5 # go from 1.0 to 0.8 to 0.5 at least (0.5 good so far)
epsilon_final = 0.1
# Calculate decay rate
decay_rate = (epsilon_initial - epsilon_final) / train_episodes


train_list_of_episodic_rewards = []
train_list_of_episodic_states = []

for episode in range(train_episodes):
#     if episode % 1000 == 0:
#         episode_reward, episode_steps, state =  0, 0, grid.reset(random_agents=True, random_default_edges=True) # start from the first node
#     else:
    episode_reward, episode_steps, state =  0, 0, grid.reset()
    
    episode_states = []
    episode_states.append(state)
    done = False
    while not done:
        neighbour_states = []
        for i in range(n_actions):
            neighbour_states.append(grid.q_get_neighbors(state[i]))
            neighbour_states[i].append(state[i])
        state_index = states_value_index[tuple(state[i] for i in range(n_actions))]
        
        neighbour_states_pairs = list(product(*neighbour_states))  
        neighbour_states_pairs_index = [states_value_index[state_pair] for state_pair in neighbour_states_pairs]
        
        epsilon_i = max(epsilon_final, epsilon_initial - decay_rate * episode)
        if np.random.uniform(0, 1) < epsilon_i:
            actions = []
            for i in range(n_actions):
                actions.append(grid.get_random_node_from_list(neighbour_states[i]))
            action = actions
            action_index = states_value_index[tuple(x for x in action)]
        else:
            q_values = Q[state_index, neighbour_states_pairs_index]
            action_index = neighbour_states_pairs_index[np.argmax(q_values)]
            action = states_pairs[action_index]
        
        next_state, reward, done = grid.step(action)   
        next_state_index = states_value_index[tuple(x for x in next_state)]
        Q[state_index, action_index] = (1 - alpha) * Q[state_index, action_index] + alpha * (reward + gamma * np.max(Q[next_state_index, :]))
        # set the new state as the current state
        episode_states.append(next_state)
        state = next_state
        
        episode_reward += reward
        if done:
            break

    if episode >= 3:
        train_list_of_episodic_rewards.append(episode_reward)
    else:
        train_list_of_episodic_rewards.append(np.nan)
    train_list_of_episodic_states.append(episode_states)
        
stop_train = time.time()
print(f"train_time = {stop_train - start_train}")

start_Q = time.time()
save_q_value(Q_values=Q)
stop_Q = time.time()
print(f"Q_time = {stop_Q - start_Q}")
grid.close()

train_episodes = np.array([i for i in range(train_episodes)]) 
train_rewards = np.array(train_list_of_episodic_rewards)    
plot_graph_test(train_episodes, train_rewards, num_nodes, grid.n_agents, type_of_graph, type="train")



start_test = time.time()
grid.reset()
grid.init_testing_params()
test_episodes = 1
test_list_of_episodic_rewards = []
test_list_of_episodic_actions = []
test_list_of_agent_positions = []
states_list = []
render = True
for episode in range(test_episodes):
    state, test_episode_reward, test_episode_steps, test_episdoe_actions = grid.reset(),0,0, []
    states_list.append(tuple(state))
    done = False
    agent_positions = []
    agent_positions.append(state)
    while not done:
        
        next_state, reward, done = grid.step_test(state)
        states_list.append(next_state)
        agent_positions.append(next_state)
        #env.render()
        state = next_state
        
        test_episode_reward += reward
        test_episode_steps += 1
        
        if done:
            break

    test_list_of_episodic_rewards.append(test_episode_reward)
    test_list_of_episodic_actions.append(test_episdoe_actions)
    test_list_of_agent_positions.append(agent_positions)
# env.save_gif('visualization/large_multiagents_q_learning_with_risk.gif')  # or env.save_mp4('output.mp4') if you want a video
grid.close()

test_episodes = np.array([i for i in range(test_episodes)]) 
test_rewards = np.array(test_list_of_episodic_rewards)
stop_test = time.time()
print(states_list)
print("Testing Time: " + str(stop_test-start_test))

if args:
    interval = int(args[0])
    app = dash.Dash(__name__)

    nav_btn_style1 = {
        'align': 'top',
        'color': 'white', 
        'backgroundColor': '#101820',
        'fontSize': '1rem',
        'width': '10rem',
        'height': '3.2rem',
        'margin': '0rem 1rem',
    }

    nav_btn_style2 = {
        'align': 'top',
        'color': 'white', 
        'backgroundColor': '#101820',
        'fontSize': '1rem',
        'width': '10rem',
        'height': '3.2rem',
        'margin': '0rem 1rem',
    }

    app.layout = html.Div([
            dcc.Interval(
                    id='interval-component',
                    interval=interval,
                    n_intervals=0
            ),
            html.Div([dcc.Graph(id='live-graph')], style={'margin-top': '25px', 'margin-left': '18px'}),
            dcc.Store(id='done', data = False),
            dcc.Store(id='flag', data = True)
        ])


    @app.callback(
        Output(component_id='live-graph', component_property='figure'),
        Output(component_id='done', component_property='data'),
        Output(component_id='flag', component_property='data'),
        [Input('interval-component', 'n_intervals'), Input("done", "data"), Input("flag", "data")]
        
    )
    def update_graph_scatter(interval, done, flag):
        if not done:
            observations, reward, done = grid.step_random()    
        elif done and flag: 
            flag = False
        data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
        return {'data': data, 'layout': layout}, done, flag

    port = 1222
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:{}".format(port))

    Timer(1, open_browser).start();
    app.run_server(debug=True, port=port)
    



















else:
    threeD = True
    app = dash.Dash(__name__)
    state, test_episode_reward, test_episode_steps, test_episdoe_actions = grid.reset(),0,0, []
    state_list = [tuple(grid.agent_positions)]
    grid.init_testing_params()

    nav_btn_style1 = {
        'align': 'top',
        'color': 'white', 
        'backgroundColor': '#101820',
        'fontSize': '1rem',
        'width': '10rem',
        'height': '3.2rem',
        'margin': '0rem 1rem',
    }

    nav_btn_style2 = {
        'align': 'top',
        'color': 'white', 
        'backgroundColor': '#101820',
        'fontSize': '1rem',
        'width': '10rem',
        'height': '3.2rem',
        'margin': '0rem 1rem',
    }

    app.layout = html.Div([
            html.Div([
                html.Div([
                    dbc.Button("Previous time-step", id="previous", style=nav_btn_style1, active=True, n_clicks=0, n_clicks_timestamp=-1),
                ], style={'display': 'inline-block', 'margin-left': '0px', 'margin-top': '10px', 'height': '10px'}),
                html.Div([
                    dbc.Button("Next time-step", id="next", style=nav_btn_style2, active=True, n_clicks=0, n_clicks_timestamp=-1)
                ], style={'display': 'inline-block', 'margin-left': '450px', 'height': '10px'})

            ]),
            html.Div([dcc.Graph(id='live-graph')], style={'margin-top': '25px', 'margin-left': '18px'}),
            dcc.Store(id='i', data=0),
            dcc.Store(id='max_i', data=0),
            dcc.Store(id='plotly_fig_table', data=[]),
            dcc.Store(id='done_i', data=99999999),
            dcc.Store(id='i_bool', data=True),
            dcc.Store(id='state', data=state),
        ])


    @app.callback(
        Output(component_id='live-graph', component_property='figure'),
        Output(component_id='i', component_property='data'),
        Output(component_id='max_i', component_property='data'),
        Output(component_id='plotly_fig_table', component_property='data'),
        Output(component_id='done_i', component_property='data'),
        Output(component_id='state', component_property='data'),
        [Input("next", "n_clicks"), Input("next", "n_clicks_timestamp"), Input("previous", "n_clicks"), Input("previous", "n_clicks_timestamp"), Input("i", "data"), Input("max_i", "data"), Input("plotly_fig_table", "data"), Input("done_i", "data"), Input("state", "data"),]
        #Input("states_pairs", "data"), Input("states_value_index", "data"), Input("q", "data")
    )
    def update_graph_scatter(n_clicks1, n_clicks_timestamp1, n_clicks2, n_clicks_timestamp2, i, max_i, plotly_fig_table, done_i, state):
        #states_pairs, states_value_index, q
        if 'next' == ctx.triggered_id:   
            if i < done_i:
                i += 1   
            if i > max_i:
                max_i = i
                next_state, reward, done = grid.step_test(state)
                state_list.append(tuple(grid.agent_positions))
                print(state_list)
                state = next_state
                if done:
                    done_i = i
                #print(i)
                data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
                #print(i)
                plotly_fig_table.append((data, layout))
            else:
                data, layout = plotly_fig_table[i]
        elif 'previous' == ctx.triggered_id:
            if i >= 1:
                i -= 1
            data, layout = plotly_fig_table[i]
        else:
            data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
            
            if i == 0 and len(plotly_fig_table) == 0:
                file_to_delete = open("adjacency.txt",'w')
                file_to_delete.close()
                plotly_fig_table.append((data, layout))
        
        return {'data': data, 'layout': layout}, i, max_i, plotly_fig_table, done_i, state

    port = 1222
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:{}".format(port))

    Timer(1, open_browser).start();
    app.run_server(debug=False, port=port)
    
        
