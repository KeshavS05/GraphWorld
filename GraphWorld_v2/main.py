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
import webbrowser
from plotly.subplots import make_subplots
from threading import Timer


grid = GraphWorld(threeD=True, length=10, height=10, depth=10, num_nodes=7, num_edges=16, fully_connected=True, n_agents=5, num_terminating_nodes=2, collaborative=True, ratio = 0.25, weight_before = 1.5, weight_after = 0.5)
grid.reset()    
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
# data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
# plotly_fig_table = [(data, layout)]
# index = 0
# max_i = 0
app.layout = html.Div([
        html.Div([
            html.Div([
                dbc.Button("Previous time-step", id="previous", style=nav_btn_style1, active=True, n_clicks=0, n_clicks_timestamp=-1),
            ], style={'display': 'inline-block', 'margin-left': '0px', 'margin-top': '10px', 'height': '10px'}),
            html.Div([
                dbc.Button("Next time-step", id="next", style=nav_btn_style2, active=True, n_clicks=0, n_clicks_timestamp=-1)
            ], style={'display': 'inline-block', 'margin-left': '450px', 'height': '10px'}),
            html.Div(str(grid.adj),
            style={'color': 'blue', 'fontSize': '1vmin', 'display': 'inline-block', 'height': '10px'})
        ]),
        html.Div([dcc.Graph(id='live-graph')], style={'margin-top': '25px', 'margin-left': '18px'}),
        dcc.Store(id='i', data = 0),
        dcc.Store(id='max_i', data = 0),
        dcc.Store(id='plotly_fig_table', data = 0),
    ])


@app.callback(
    Output(component_id='live-graph', component_property='figure'),
    [Input("next", "n_clicks"), Input("next", "n_clicks_timestamp"), Input("previous", "n_clicks"), Input("previous", "n_clicks_timestamp"), Input('i')]
    
)
def update_graph_scatter(n_clicks1, n_clicks_timestamp1, n_clicks2, n_clicks_timestamp2):  
    print(')
    if 'next' == ctx.triggered_id:
        i += 1
        if i > max_i:
            max_i = i
            grid.step()
        data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
        plotly_fig_table.append((data, layout))
        return {'data': data,
                'layout': layout}
    elif 'previous' == ctx.triggered_id:
        print("prev")
        if i >= 1:
            i -= 1
        data, layout = plotly_fig_table[i]
        return {'data': data,
                'layout': layout}
    else:
        data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
        if i == 0 and len(plotly_fig_table) == 0:
            plotly_graph_table = [(data, layout)]
        return {'data': data,
                'layout': layout}
    #     data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
    # if 'next' == ctx.triggered_id:
    #     #data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
    #     i += 1
    #     #if i > max_i:
    #     max_i = i
    #     grid.step()
    #     data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
    #     plotly_fig_table.append((data, layout))
    #     #else:
    #         #data, layout = plotly_fig_table[i][0], plotly_fig_table[i][1]
    # elif 'previous' == ctx.triggered_id:
    #     if i >= 1:
    #         i -= 1
    #     print('hi')
    #     #data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
    #     data, layout = plotly_fig_table[i][0], plotly_fig_table[i][1]
    # else:
    #     data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
    # return {'data': data,
    #         'layout': layout} 

port = 1222
def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://localhost:{}".format(port))

Timer(1, open_browser).start();
app.run_server(debug=True, port=port)
