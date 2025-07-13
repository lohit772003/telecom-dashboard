import math, random
import pandas as pd, networkx as nx
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_cytoscape as cyto
import os

# --- Load Data ---
devices = pd.read_csv("Devices.csv")
edges = pd.read_csv("DeviceHasInterface.csv")
edges['from_key'] = edges['_from'].str.split('/').str[-1]
edges['to_key'] = edges['_to'].str.split('/').str[-1]

# --- Handover Reliability ---
ho_df = pd.concat([
    edges[['from_key','calls_done','calls_accepted']].rename(columns={'from_key':'_key'}),
    edges[['to_key','calls_done','calls_accepted']].rename(columns={'to_key':'_key'})
])
agg = ho_df.groupby('_key').sum(numeric_only=True)
agg['handover_rel'] = (agg['calls_accepted'] / agg['calls_done'].replace({0: pd.NA})).round(2)
devices = devices.merge(agg[['handover_rel']], on='_key', how='left')

# --- Status Classification ---
mean_calls = devices.groupby('device_type')['calls'].mean()
dev_calls = devices.groupby('device_type')['calls'].transform(lambda x: (x - x.mean()).abs().mean())

def classify(row):
    m = mean_calls.get(row['device_type'], 0)
    d = dev_calls.get(row.name, 0)
    if pd.notna(row['handover_rel']) and row['handover_rel'] < 0.95:
        return 'PoorHO'
    elif row['calls'] > m + d:
        return 'Congested'
    elif row['calls'] < m - d:
        return 'Weak'
    return 'Normal'

devices['status'] = devices.apply(classify, axis=1)
status_by_key = devices.set_index('_key')['status'].to_dict()

# --- Build Graph ---
G = nx.Graph()
for _, r in devices.iterrows():
    G.add_node(r['_key'], **r.to_dict())
for _, r in edges.iterrows():
    if r['from_key'] in G and r['to_key'] in G:
        G.add_edge(r['from_key'], r['to_key'])

areas = sorted(devices['area_id'].unique())
area_type_map = {a: 'Urban' if i % 2 == 0 else 'Rural' for i, a in enumerate(areas)}

# --- Colors ---
area_color = {
    a: f"hsl({i * 60 % 360}, 70%, 60%)" for i, a in enumerate(areas)
}
status_color = {
    'Normal':'#3498db', 'Congested':'#e74c3c',
    'Weak':'#f39c12', 'PoorHO':'#8e44ad'
}

stylesheet = [
    {'selector': 'node', 'style': {
        'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center',
        'color': 'white', 'font-size': '10px', 'width': 30, 'height': 30
    }},
    {'selector': 'edge', 'style': {'line-color': '#c0c0c0', 'width': 1}}
] + [
    {'selector': f'.{a}', 'style': {'background-color': c}} for a, c in area_color.items()
] + [
    {'selector': f'.{s}', 'style': {'background-color': c}} for s, c in status_color.items()
]

# --- App Init ---
app = Dash(__name__)
app.title = "📡 Telecom Network Dashboard"

# --- Legend UI ---
def make_legend():
    return html.Div([
        html.Div([
            html.Strong('Area:'), *[
                html.Div([
                    html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px',
                                     'backgroundColor': c, 'marginRight': '4px'}), a
                ]) for a, c in area_color.items()
            ]
        ], id='area-legend', style={'marginRight': '24px', 'display': 'flex'}),
        html.Div([
            html.Strong('Status:'), *[
                html.Div([
                    html.Span(style={'display': 'inline-block', 'width': '12px', 'height': '12px',
                                     'backgroundColor': c, 'marginRight': '4px'}), s
                ]) for s, c in status_color.items()
            ]
        ], id='status-legend', style={'display': 'none'})
    ], style={'display': 'flex', 'justifyContent': 'flex-end', 'gap': '32px', 'paddingRight': '24px'})

# Create ticket log file if not exists
log_file = "ticket_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=['Area', 'Device', 'Type', 'Status', 'Suggestion']).to_csv(log_file, index=False)

# --- Layout ---
app.layout = html.Div([
    html.H3("📡 Telecom Network Dashboard", style={'textAlign': 'center', 'marginBottom': '8px'}),
    make_legend(),
    html.Div([
        html.Button("🔍 Scan Network", id='scan', n_clicks=0, style={'marginRight': '12px'}),
        html.Button("🚨 Raise Ticket", id='ticket', n_clicks=0, disabled=True)
    ], style={'textAlign': 'center'}),
    dcc.Dropdown(id='area', options=[{'label': a, 'value': a} for a in areas],
                 placeholder="Select Area", style={'width': '40%', 'margin': '10px auto'}),
    html.Div(id='area-type', style={'textAlign': 'center', 'fontWeight': 'bold'}),
    html.Div(id='scan-out', style={'textAlign': 'center'}),
    html.Div([
        cyto.Cytoscape(id='graph', elements=[], stylesheet=stylesheet,
                       layout={'name': 'preset'}, style={'width': '70%', 'height': '520px'}),
        html.Div(id='table-container', style={'width': '28%', 'paddingLeft': '1%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    html.Div(id='hover', style={'textAlign': 'center', 'marginTop': '10px'}),
    html.H4("📋 Raised Ticket Table", style={'textAlign': 'center'}),
    html.Div(id='ticket-log', style={'width': '90%', 'margin': '0 auto'})
])

# --- Callback: Area selection ---
@app.callback(
    Output('graph', 'elements'),
    Output('graph', 'layout'),
    Output('table-container', 'children'),
    Output('ticket', 'disabled'),
    Output('area-legend', 'style'),
    Output('status-legend', 'style'),
    Input('area', 'value')
)
def update_graph(area):
    if not area:
        elems = []
        center = {}
        R = 400
        for i, a in enumerate(areas):
            angle = 2 * math.pi * i / len(areas)
            center[a] = (R * math.cos(angle), R * math.sin(angle))
        for _, r in devices.iterrows():
            cx, cy = center[r['area_id']]
            elems.append({
                'data': {'id': r['_key'], 'label': '', 'tooltip': r['_key']},
                'position': {'x': cx + random.uniform(-80, 80), 'y': cy + random.uniform(-80, 80)},
                'classes': r['area_id']
            })
        for u, v in G.edges():
            elems.append({'data': {'id': f"{u}_{v}", 'source': u, 'target': v}})
        return elems, {'name': 'preset', 'fit': True}, '', True, {'display': 'flex'}, {'display': 'none'}

    df = devices[devices['area_id'] == area].copy().reset_index(drop=True)
    elems = []
    r = 200
    for i, row in df.iterrows():
        angle = 2 * math.pi * i / len(df)
        x, y = r * math.cos(angle), r * math.sin(angle)
        tip = f"{row['hostname']} | {row['device_type']} | Calls: {row['calls']} | HOrel: {row.get('handover_rel', '')} | {row['status']}"
        elems.append({
            'data': {'id': row['_key'], 'label': str(i+1), 'tooltip': tip},
            'position': {'x': x, 'y': y},
            'classes': row['status']
        })
    for u, v in G.edges():
        if u in df['_key'].values and v in df['_key'].values:
            elems.append({'data': {'id': f"{u}_{v}", 'source': u, 'target': v}})

    table = dash_table.DataTable(
        columns=[
            {'name': 'Node #', 'id': 'number'},
            {'name': 'Device Type', 'id': 'device_type'},
            {'name': 'Calls', 'id': 'calls'},
            {'name': 'HO Reliability', 'id': 'handover_rel'}
        ],
        data=[{
            'number': i+1,
            'device_type': row['device_type'],
            'calls': row['calls'],
            'handover_rel': row.get('handover_rel', '')
        } for i, row in df.iterrows()],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '6px'},
        style_header={'backgroundColor': '#eaf2f8', 'fontWeight': 'bold'},
        style_data={'backgroundColor': '#fcfcfc'}
    )
    return elems, {'name': 'preset', 'fit': True}, table, False, {'display': 'none'}, {'display': 'flex'}

@app.callback(
    Output('area-type', 'children'),
    Input('area', 'value')
)
def show_area_type(area):
    return f"🗺️ This area is classified as: {area_type_map.get(area, 'Unknown')}" if area else ''

@app.callback(
    Output('hover', 'children'),
    Input('graph', 'mouseoverNodeData')
)
def hover_info(data):
    return f"🔎 {data['tooltip']}" if data else ''

@app.callback(
    Output('scan-out', 'children'),
    Input('scan', 'n_clicks')
)
def scan_network(n):
    if n == 0: return ''
    messages = []
    for area in areas:
        sub = devices[devices['area_id'] == area]
        issues = sub[sub['status'].isin(['Congested','Weak','PoorHO'])]
        if not issues.empty:
            issue_counts = issues['status'].value_counts().to_dict()
            issue_msg = ', '.join([f"{v} {k}" for k, v in issue_counts.items()])
            messages.append(f"⚠️ {area} has issues: {issue_msg}")
    return html.Div([html.Div(m) for m in messages]) if messages else "✅ No issues found."

@app.callback(
    Output('ticket-log', 'children'),
    Input('ticket', 'n_clicks'),
    State('area', 'value')
)
def raise_ticket(n, area):
    if not n or not area:
        return dash_table.DataTable(columns=[], data=[])
    issues = devices[(devices['area_id'] == area) & (devices['status'].isin(['Congested','Weak','PoorHO']))]
    rows = []
    for _, row in issues.iterrows():
        suggestion = suggest(row['device_type'], row['status'])
        rows.append({
            'Area': area,
            'Device': row['_key'],
            'Type': row['device_type'],
            'Status': row['status'],
            'Suggestion': suggestion
        })
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True)
    log_df.to_csv(log_file, index=False)
    return dash_table.DataTable(
        columns=[{'name': i, 'id': i} for i in ['Area', 'Device', 'Type', 'Status', 'Suggestion']],
        data=log_df.to_dict('records'),
        style_table={'overflowX': 'auto', 'marginTop': '10px'},
        style_cell={'textAlign': 'center', 'padding': '6px'},
        style_header={'backgroundColor': '#eaf2f8', 'fontWeight': 'bold'},
        style_data={'backgroundColor': '#fcfcfc'}
    )

def suggest(dev_type, status):
    if status == 'PoorHO': return 'Check handover params'
    if status == 'Congested': return f"Add capacity to {dev_type}"
    if status == 'Weak': return f"Check link or replace {dev_type}"
    return 'No action'

# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)
