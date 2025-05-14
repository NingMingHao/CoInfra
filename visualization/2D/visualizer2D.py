import os
import yaml
import numpy as np
import cv2
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import glob

class CoInfraVisualizer:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.TEST_FOLDER = None
        self.frame_list = []
        self.cached_obstacles = {}
        self.roi_map_dict = {}
        self.ground_to_global_dict = {}
        self.hd_map_img = None
        self.img_size = None
        self.scale = None
        self.min_x = None
        self.max_y = None

        self.INTEREST_NODES = ["4", "5", "6", "7", "8", "9", "10", "11", "global"]
        self.NODE_COLORS = {
            "4": (255, 0, 0), "5": (0, 255, 0), "6": (0, 0, 255), "7": (255, 255, 0),
            "8": (0, 255, 255), "9": (255, 0, 255), "10": (128, 128, 0), "11": (0, 128, 128), "global": (128, 0, 128)
        }

        self.setup_layout()

    def load_data_from_folder(self, folder):
        self.TEST_FOLDER = folder
        transformation_yaml_path = os.path.join(folder, "HDmap", "transformation.yaml")
        with open(transformation_yaml_path, 'r') as f:
            transformation_info = yaml.safe_load(f)

        self.img_size = transformation_info["img_size"]
        self.scale = transformation_info["scale"]
        self.min_x = transformation_info["min_x"]
        self.max_y = transformation_info["max_y"]

        self.roi_map_dict = {}
        for nid in self.INTEREST_NODES:
            roi_path = os.path.join(folder, "HDmap", "ROI", f"{nid}_roi.png")
            self.roi_map_dict[nid] = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        self.hd_map_img = cv2.imread(os.path.join(folder, "HDmap", "hdmap.png"))

        self.ground_to_global_dict = {}
        for node_id in self.INTEREST_NODES:
            if node_id == "global":
                self.ground_to_global_dict[node_id] = np.eye(4)
            else:
                tf_path = os.path.join(folder, "Calibration", f"lidar/{node_id}.yaml")
                with open(tf_path, 'r') as f:
                    tf_info = yaml.safe_load(f)
                self.ground_to_global_dict[node_id] = np.array(tf_info['ground_to_global'])

        gt_folder = os.path.join(folder, "GroundTruth")
        self.frame_list = sorted([f for f in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, f))])
        self.cached_obstacles = {
            frame: {
                node: self.load_yaml_obstacles(os.path.join(gt_folder, frame), node)
                for node in self.INTEREST_NODES
            } for frame in self.frame_list
        }

    def load_yaml_obstacles(self, frame_folder, node_id):
        path = os.path.join(frame_folder, f"{node_id}_roi.yaml")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return []

    def transform_to_global(self, pos, heading, transform_mat):
        local_vec = np.array([pos['x'], pos['y'], pos['z'], 1.0])
        global_vec = transform_mat @ local_vec
        new_pos = {'x': float(global_vec[0]), 'y': float(global_vec[1]), 'z': float(global_vec[2])}
        R = transform_mat[:3, :3]
        yaw_offset = np.arctan2(R[1, 0], R[0, 0])
        global_heading = heading + yaw_offset
        return new_pos, ((global_heading + np.pi) % (2 * np.pi)) - np.pi

    def draw_bbox_on_fig(self, fig, obj_list, color='red', node_id='', global_mode=True, transform=None):
        for obj in obj_list:
            pos = obj['position']
            heading = obj['heading']
            if not global_mode and transform is not None:
                pos, heading = self.transform_to_global(pos, heading, transform)

            l, w = obj['dimensions']['length'], obj['dimensions']['width']
            cx, cy = pos['x'], pos['y']
            dx, dy = l / 2.0, w / 2.0
            corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy], [dx, dy]])
            rot = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
            rotated = (rot @ corners.T).T + np.array([cx, cy])
            x, y = rotated[:, 0], rotated[:, 1]
            node_str = f"({node_id})" if node_id != "global" else "(gl)"
            name = f"{obj['id']}-{obj['label']}" if global_mode else f"{obj['id']}-{obj['label']}" + node_str
            hover_text = f"ID: {obj['id']}<br>Label: {obj['label']}<br>Length: {l:.2f}<br>Width: {w:.2f}<br>Heading: {heading:.2f}"

            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=5), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(
                color=color, width=2), name=name, text=[hover_text]*len(x), hoverinfo='text'))

            arrow_len = 0.6 * l
            arrow_end = [cx + arrow_len * np.cos(heading), cy + arrow_len * np.sin(heading)]
            fig.add_trace(go.Scatter(x=[cx, arrow_end[0]], y=[cy, arrow_end[1]], mode='lines+markers', line=dict(color=color, width=2, dash='dash'), marker=dict(size=2), showlegend=False, hoverinfo='skip'))

    def get_available_folders(self, root_dir):
        folders = [f for f in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(f)]
        # print(f"Available folders: {folders}")
        return [{'label': os.path.basename(f), 'value': f} for f in folders]

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H2("CoInfra 2D ROI & Ground Truth Viewer"),

            # Frame Selection Row
            html.Div([
                html.Div([
                    html.Label("Select Frame:"),
                    dcc.Slider(id='frame-slider', step=1, value=0,
                               tooltip={"placement": "bottom"})
                ], style={'width': '85%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '10px'}),
                html.Div([
                    html.Button('Play', id='play-button', n_clicks=0),
                    dcc.Interval(id='play-interval', interval=500,
                                 n_intervals=0, disabled=True)
                ], style={'display': 'inline-block', 'verticalAlign': 'top', 'paddingTop': '30px'})
            ], style={'paddingBottom': '20px'}),

            # Controls Row
            html.Div([
                html.Div([
                    html.Label("Base Layer:"),
                    dcc.RadioItems(id='base-layer', options=[
                        {'label': 'HD Map', 'value': 'hdmap'},
                        {'label': 'ROI Map', 'value': 'roi'}
                    ], value='roi'),

                    html.Label("Mode:"),
                    dcc.RadioItems(id='view-mode', options=[
                        {'label': 'Global Mode', 'value': 'global'},
                        {'label': 'Individual Node Mode', 'value': 'node'}
                    ], value='global'),

                    html.Label("Enable Nodes (for Individual Mode):"),
                    dcc.Checklist(id='node-selection',
                                  inline=True, value=['global']),

                    html.Label("Select Folder:"),
                    dcc.Dropdown(id='folder-path',
                                 placeholder='Select a dataset folder'),
                    html.Button('Load Folder', id='load-folder', n_clicks=0)
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    dcc.Graph(id='roi-graph')
                ], style={'width': '69%', 'display': 'inline-block'})
            ])
        ])

        @self.app.callback(Output('folder-path', 'options'), Input('load-folder', 'n_clicks'), prevent_initial_call=True)
        def update_dropdown(_):
            return self.get_available_folders(os.path.expanduser('~/Documents/Gits/OutdoorSensorNodes/CoInfra/example_data'))

        @self.app.callback(Output('play-interval', 'disabled'), Input('play-button', 'n_clicks'), prevent_initial_call=True)
        def toggle_play(n):
            return n % 2 == 0

        @self.app.callback(
            Output('frame-slider', 'value'),
            Input('play-interval', 'n_intervals'),
            State('frame-slider', 'value')
        )
        def advance_frame(_, current):
            if not self.frame_list:
                return 0
            return (current + 1) % len(self.frame_list)


        @self.app.callback(
            [Output('roi-graph', 'figure'),
             Output('frame-slider', 'max'),
             Output('frame-slider', 'marks'),
             Output('node-selection', 'options')],
            [Input('frame-slider', 'value'),
             Input('base-layer', 'value'),
             Input('view-mode', 'value'),
             Input('node-selection', 'value'),
             Input('load-folder', 'n_clicks')],
            [State('folder-path', 'value')]
        )
        def update_fig(frame_idx, base_layer, view_mode, enabled_nodes, _, folder):
            if folder and os.path.exists(folder) and folder != self.TEST_FOLDER:
                print("Reloading data from folder:", folder)
                self.load_data_from_folder(folder)

            if not self.frame_list:
                return go.Figure(), 0, {}, []

            frame = self.frame_list[frame_idx % len(self.frame_list)]
            base = np.zeros((self.img_size, self.img_size, 4), dtype=np.uint8)
            if base_layer == 'roi':
                nodes = enabled_nodes if view_mode == 'node' else ['global']
                for nid in nodes:
                    mask = self.roi_map_dict[nid] > 0
                    color = self.NODE_COLORS.get(nid, (255, 255, 255))
                    for i in range(3):
                        base[..., i][mask] = color[i]
                    base[..., 3][mask] = 100
            else:
                base[..., :3] = cv2.resize(cv2.cvtColor(self.hd_map_img, cv2.COLOR_BGR2RGB), (self.img_size, self.img_size))
                base[..., 3] = 255

            base = np.flipud(base)
            fig = go.Figure()
            fig.add_trace(go.Image(z=base, x0=self.min_x, y0=self.max_y - self.img_size/self.scale, dx=1/self.scale, dy=1/self.scale))

            if view_mode == 'global':
                self.draw_bbox_on_fig(fig, self.cached_obstacles[frame]['global'], color='red', node_id='global', global_mode=True)
            else:
                for i, nid in enumerate(enabled_nodes):
                    obj_list = self.cached_obstacles[frame][nid]
                    color = f"hsl({(i * 40) % 360},100%,50%)"
                    tf = self.ground_to_global_dict[nid]
                    self.draw_bbox_on_fig(fig, obj_list, color=color, node_id=nid, global_mode=False, transform=tf)

            fig.update_xaxes(title_text="X (m)", range=[self.min_x, self.min_x + self.img_size/self.scale], constrain='domain')
            fig.update_yaxes(title_text="Y (m)", range=[self.max_y - self.img_size/self.scale, self.max_y], scaleanchor='x', constrain='domain')
            fig.update_layout(height=800, width=800, hovermode='closest', autosize=False, margin=dict(l=10, r=10, t=40, b=40), showlegend=False)
            return fig, len(self.frame_list)-1, {i: f if i % 10 == 0 else '' for i, f in enumerate(self.frame_list)}, [{'label': nid, 'value': nid} for nid in self.INTEREST_NODES]

    def run(self):
        self.app.run(debug=False)

if __name__ == '__main__':
    viewer = CoInfraVisualizer()
    viewer.run()
