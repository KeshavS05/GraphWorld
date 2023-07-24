import math
import numpy as np
import random
import cv2
from GraphWorld import GraphWorld
from threeD_render import threeD_Render
import time
import plotly.graph_objs as go
import plotly
import dash
from dash import ctx
from dash.dependencies import Output, Input, State
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import os
import webbrowser, pyautogui
from threading import Timer
import sys; args = sys.argv[1:]
import atexit

threeD = True
grid = GraphWorld(threeD=True, html=True, length=20, height=20, depth=20, num_nodes=10, num_edges=20, fully_connected=True, n_agents=5, num_terminating_nodes=2, collaborative=True, ratio = 0.1, weight_before = 1.5, weight_after = 0.1)
grid.reset() 

def write_to_file(i):
    with open('adjacency.txt', 'a') as f:
        if i == 0:
            f.write('Non-terminal node labels: ')
            supporting_nodes = []
            for i2, d_node in enumerate(grid.non_terminal_nodes):
                if (n := grid.get_node(d_node)).supporting:
                    supporting_nodes.append(d_node)
                f.write(str(d_node))
                if i2 != len(grid.non_terminal_nodes):
                    f.write(', ')
            f.write('\n')
            f.write('Terminal node labels: ')
            for i2, t_node in enumerate(grid.terminal_nodes):
                if (n := grid.get_node(t_node)).supporting:
                    supporting_nodes.append(t_node)
                f.write(str(t_node))
                if i2 != len(grid.terminal_nodes):
                    f.write(', ')
            f.write('\n')
            f.write('Supporting node labels: ')
            for i2, s_node in enumerate(supporting_nodes):
                f.write(str(s_node))
                if i2 != len(supporting_nodes):
                    f.write(', ')  
            f.write('\n\n\n')       
                    
                    
        f.write("Time step: " + str(i) + '\n')
        f.write(grid.adj_to_str())
        f.write('\n')
        for a in grid.agents:
            f.write(f'Agent {a.label} is on node {a.node.node_label}. Agent {a.label} is on a terminating node: {a.node.terminating} \n')
        f.write('\n\n')
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
            observations, reward, done = grid.step()    
            write_to_file(grid.time_step)  
        elif done and flag:
            write_to_file(grid.time_step) 
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
        ])


    @app.callback(
        Output(component_id='live-graph', component_property='figure'),
        Output(component_id='i', component_property='data'),
        Output(component_id='max_i', component_property='data'),
        Output(component_id='plotly_fig_table', component_property='data'),
        Output(component_id='done_i', component_property='data'),
        [Input("next", "n_clicks"), Input("next", "n_clicks_timestamp"), Input("previous", "n_clicks"), Input("previous", "n_clicks_timestamp"), Input("i", "data"), Input("max_i", "data"), Input("plotly_fig_table", "data"), Input("done_i", "data")]
        
    )
    def update_graph_scatter(n_clicks1, n_clicks_timestamp1, n_clicks2, n_clicks_timestamp2, i, max_i, plotly_fig_table, done_i):
        if 'next' == ctx.triggered_id:   
            if i < done_i:
                i += 1   
            if i > max_i:
                max_i = i
                observations, reward, done = grid.step()
                write_to_file(i)
                if done:
                    done_i = i
                data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
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
                write_to_file(i)
                plotly_fig_table.append((data, layout))
        
        return {'data': data, 'layout': layout}, i, max_i, plotly_fig_table, done_i

    port = 1222
    def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://localhost:{}".format(port))

    Timer(1, open_browser).start();
    app.run_server(debug=True, port=port)
    
        
