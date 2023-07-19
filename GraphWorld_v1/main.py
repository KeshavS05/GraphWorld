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


grid = GraphWorld(threeD=True, length=10, height=10, depth=10, num_nodes=7, num_edges=16, fully_connected=True, n_agents=5, num_terminating_nodes=2, collaborative=True, ratio = 0.25, weight_before = 1.5, weight_after = 0.5)
grid.reset()    
app = dash.Dash(__name__)

app.layout = html.Div([
        dcc.Graph(id='live-graph'),
        html.Div([
            dbc.Button("Previous time-step", id="previous", className="me-2", active=True, n_clicks=0, n_clicks_timestamp=-1),
            dbc.Button("Next time-step", id="button", size="lg", className="me-1", color="primary", active=True, n_clicks=0, n_clicks_timestamp=-1)
        ])
    ])


@app.callback(
    Output(component_id='live-graph', component_property='figure'),
    [Input("button", "n_clicks"), Input("button", "n_clicks_timestamp")]
    
)
def update_graph_scatter(n_clicks, n_clicks_timestamp):
    if n_clicks >= 1:
        grid.step()
        data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
        return {'data': data,
                'layout': layout}
    else:
        data, layout = grid.render(grid_line_color=(0, 0, 255), grid_tick_val=int(grid.length/10), default_edge_color=(0, 255, 0), active_support_edge_color=(0, 165, 255), inactive_support_edge_color=(128, 0, 128), node_color=(255, 255, 255), node_border_color=(203, 192, 255), terminating_node_color=(0, 0, 255), supporting_node_color=(255, 0, 0), agent_color=(255, 255, 0), agent_label_text_color=(0, 0, 0))
        return {'data': data,
                'layout': layout}

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:1222/')

app.run_server(debug=False, use_reloader=False, port=1222)
