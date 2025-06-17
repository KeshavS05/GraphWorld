import plotly.graph_objs as go
import plotly

class threeD_Render():
    
    def __init__(self, length, height, depth, G, agents, type_of_weighted, supporting_edge_dict) -> None:
        self.G = G
        self.length = length
        self.height = height
        self.depth = depth
        self.nodes = self.G.nodes
        self.edges = self.G.edges(data="weight")
        self.agents = agents
        self.type_of_weighted = type_of_weighted
        self.supporting_edge_dict = supporting_edge_dict
        
    def get_node(self, node_label):
        return self.nodes[node_label]["val"]

    def color_string(self, color):
        return 'rgb(' + str(color[2]) + ', ' + str(color[1]) + ', ' + str(color[0]) + ')'

    def rgb_to_hex(self, r, g, b):
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    def render(self, grid_line_color, grid_tick_val, default_edge_color, active_support_edge_color, inactive_support_edge_color, node_color, terminating_node_color, supporting_node_color, agent_color, agent_label_text_color): 
        scale_factor = int(800/self.length)
        dxN, dyN, dzN = [], [], []
        sxN, syN, szN = [], [], []
        txN, tyN, tzN = [], [], []
        axN, ayN, azN = [], [], []
        agent_labels = []
    

        dxE, dyE, dzE = [], [], []
        bxE, byE, bzE = [], [], []
        axE, ayE, azE = [], [], []
        
        
        xA, yA, zA = [], [], []
        an = []
        
        weight = []
        node_label_d = []
        node_label_s = []
        node_label_t = []
        b_support = []
        a_support = []
        d_edge = []
        
        node_agents = {} 
        for a in self.agents:
            coords = a.node.return_coords()
            if a.node.node_label in node_agents:
                node_agents[a.node.node_label].append(a.label)
            else:
                node_agents[a.node.node_label] = [a.label]
            xA.append(coords[0])
            yA.append(coords[1])
            zA.append(coords[2])
            # an.append(a.node.node_label)

        for n_label in self.nodes:
            n = self.get_node(n_label)
            coords = n.return_coords()
            if n_label in node_agents:
                s = '['
                for i, a in enumerate(node_agents[n_label]):
                    s += str(a)
                    if i != len(node_agents[n_label]) - 1:
                        s += ','
                s += ']'
                axN.append(coords[0])
                ayN.append(coords[1])
                azN.append(coords[2])
                agent_labels.append('<b>' + s + '</b>')

            if n.terminating:
                txN.append(coords[0])
                tyN.append(coords[1])
                tzN.append(coords[2])
                node_label_t.append(n_label)
            elif n.supporting:
                sxN.append(coords[0])
                syN.append(coords[1])
                szN.append(coords[2])
                node_label_s.append(n_label)
            else:
                dxN.append(coords[0])
                dyN.append(coords[1])
                dzN.append(coords[2])
                node_label_d.append(n_label)
        
        
        for one, two, w in self.edges:
            node_one_coords = self.get_node(one).return_coords()
            node_two_coords = self.get_node(two).return_coords()
            if self.type_of_weighted[(one, two)] == 1:#w == self.weight_before:
                bxE += [node_one_coords[0], node_two_coords[0], None]
                byE += [node_one_coords[1], node_two_coords[1], None]
                bzE += [node_one_coords[2], node_two_coords[2], None]
                sup_nodes = self.supporting_edge_dict[(one, two)]
                inp = f"Edge between Node {one} and Node {two}. Supporting nodes: {sup_nodes}"
                b_support.append(inp)
            elif self.type_of_weighted[(one, two)] == 2:#w == self.weight_after:
                axE += [node_one_coords[0], node_two_coords[0], None]
                ayE += [node_one_coords[1], node_two_coords[1], None]
                azE += [node_one_coords[2], node_two_coords[2], None]
                sup_nodes = self.supporting_edge_dict[(one, two)]
                inp = f"Edge between Node {one} and Node {two}. Supporting nodes: {sup_nodes}"
                a_support.append(inp)
            elif self.type_of_weighted[(one, two)] == 0:
                dxE += [node_one_coords[0], node_two_coords[0], None]
                dyE += [node_one_coords[1], node_two_coords[1], None]
                dzE += [node_one_coords[2], node_two_coords[2], None]
                inp = f"Edge between Node {one} and Node {two}."
                d_edge.append(inp)
            weight.append(w)
        
        trace1=go.Scatter3d(x=dxN,
                    y=dyN,
                    z=dzN,
                    mode='markers',
                    name='Default Nodes',
                    marker=dict(symbol='circle',
                                    size=6,
                                    color=self.color_string(node_color),
                                    colorscale='Viridis'
                                    ),
                    text = node_label_d,
                    hoverinfo='text',
                    legendrank=1
                    )
        
        trace2=go.Scatter3d(x=sxN,
                    y=syN,
                    z=szN,
                    mode='markers',
                    name='Supporting Nodes',
                    marker=dict(symbol='circle',
                                    size=6,
                                    color="#00FFFF",
                                    colorscale='Viridis'
                                    ),
                    text = node_label_s,
                    hoverinfo='text'
                    )
        
        trace3=go.Scatter3d(x=txN,
                    y=tyN,
                    z=tzN,
                    mode='markers',
                    name='Terminating Nodes',
                    marker=dict(symbol='circle',
                                    size=15,
                                    color=self.color_string(terminating_node_color),
                                    colorscale='Viridis'
                                    ),
                    text = node_label_t,
                    hoverinfo='text'
                    )
        
        trace4=go.Scatter3d(x=dxE,
                    y=dyE,
                    z=dzE,
                    mode='lines',
                    name = "Default Edges",
                    line=dict(color=self.color_string(default_edge_color), width=2),
                    text = d_edge,
                    hoverinfo='text'
                    )
        trace5=go.Scatter3d(x=bxE,
                    y=byE,
                    z=bzE,
                    mode='lines',
                    name = "Weighted Edges (Before)",
                    line=dict(color="#FC1CBF", width=2),
                    text = b_support,
                    hoverinfo='text'
                    )
        
        trace6=go.Scatter3d(x=axE,
                    y=ayE,
                    z=azE,
                    mode='lines',
                    name = "Weighted Edges (After)",
                    line=dict(color=self.color_string(active_support_edge_color), width=2),
                    text = a_support,
                    hoverinfo='text'
                    )
        
        trace7=go.Scatter3d(x=xA,
                    y=yA,
                    z=zA,
                    mode='markers',
                    name='Agents',
                    marker=dict(symbol='square',
                                    size=15,
                                    color="#0000FF",
                                    colorscale='Viridis',
                                    )
                    
                    # text = an,
                    # hoverinfo='text'
                    )
        
        trace8=go.Scatter3d(x=axN,
                    y=ayN,
                    z=azN,
                    mode='text',
                    name='Agent Labels',
                    text=agent_labels,
                    textposition='middle center',
                    textfont = dict(family="Courier New, monospace", size=15, color='white'),
                    # hoverinfo='text'
                    )
        
        

        x_axis=dict(showbackground=True,
                showline=True,
                zeroline=False,
                showgrid=True,
                showticklabels=True,
                autorange=False,
                gridcolor='gray',#self.rgb_to_hex(grid_line_color[0], grid_line_color[1], grid_line_color[2]),
                color='#000',
                tickmode="array",
                range = [0, self.length],
                tickvals=[i for i in range(0, self.length, grid_tick_val)],
                tickfont=dict(family='Rockwell', color='white', size=14),
                title='',
                backgroundcolor="rgba(0, 0, 0,0)",
                )
        
        y_axis=dict(showbackground=True,
                showline=True,
                zeroline=False,
                showgrid=True,
                showticklabels=True,
                autorange=False,
                gridcolor="gray",
                color='#000',
                tickmode="array",
                range = [0, self.height],
                tickvals=[i for i in range(0, self.height, grid_tick_val)],
                tickfont=dict(family='Rockwell', color='white', size=14),
                title='',
                backgroundcolor="rgba(0, 0, 0, 0)",
                )
        
        z_axis=dict(showbackground=True,
                showline=True,
                zeroline=False,
                showgrid=True,
                showticklabels=True,
                autorange=False,
                gridcolor="gray",
                color='#000',
                tickmode="array",
                range = [0, self.depth],
                tickvals=[i for i in range(0, self.depth, grid_tick_val)],
                tickfont=dict(family='Rockwell', color='white', size=14),
                title='',
                backgroundcolor="rgba(0, 0, 0, 0)",
                )

        layout = go.Layout(
                width=self.length * scale_factor,
                height=self.height * scale_factor,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=10,
                    font=dict(
                        family="Courier",
                        size=20,
                        color="white"
                    ),
                    xanchor="right",
                    x=1
                    
                ),
                showlegend=True,
                paper_bgcolor ="rgba(0,0,0,255)",
                scene=dict(
                    xaxis=dict(x_axis),
                    yaxis=dict(y_axis),
                    zaxis=dict(z_axis),
                ),
                #plot_bgcolor = "black"
                
            hovermode='closest')
        data = [trace4, trace5, trace6, trace1, trace2, trace3, trace7, trace8]
        fig = go.Figure(data=data, layout=layout)
        # self.app.layout = html.Div([
        #     dcc.Graph(figure=fig)
        # ])
        #fig.show()
        # plotly.offline.plot(fig, kind="scatter")
        #return fig
        return data, layout
        #return fig
        #return fig
    