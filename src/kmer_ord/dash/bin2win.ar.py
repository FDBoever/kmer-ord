import sys
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, MATCH
import dash_bootstrap_components as dbc
import pandas as pd
import os
import sqlite3
import plotly.graph_objects as go
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import numpy as np
from dash import dash_table
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
import bokeh.palettes
from colorcet import glasbey
import io
from PIL import Image
import argparse
from math import ceil
from matplotlib.path import Path


def print_banner(host="127.0.0.1", port=8050):
    title = "BIN2WIN"
    subtitle = "INTERACTIVE BINNER v1.0"

    padding = 4
    inner_width = len(title) + padding * 2
    border = "#" * (inner_width + 2)  # +2 for the side '#'

    line_title = "#{}#".format(
        (" " * padding + title + " " * padding).center(inner_width)
    )

    lines = []
    lines.append("\n" * 2)
    lines.append(border)
    lines.append(line_title)
    lines.append(border)
    lines.append(subtitle.center(len(border)))
    lines.append("")
    lines.append(f"Please view the GUI at http://{host}:{port}/")
    lines.append("")

    banner = "\n".join(lines) + "\n"

    # Force-write and flush in case stdout is buffered
    sys.stdout.write(banner)
    sys.stdout.flush()

##############################
# ARGPARSE
##############################
parser = argparse.ArgumentParser(
    description="INTERACTIVE KMER-BASED READ ORDINATION INTERFACE - ***BIN2WIN***"
)
parser.add_argument("-d", "--database", type=str, required=True,
                    help="Path to the main features database file")
parser.add_argument("-o", "--output_dir", type=str, default="bins",
                    help="Path to output directory (default: ./bins)")
args = parser.parse_args()

# Print banner as soon as the script starts with proper args
print_banner(host="127.0.0.1", port=8050)

db_path = args.database
output_dir_default = args.output_dir


##############################
# UTILS
##############################
def get_features_columns(db_path):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(features);")
        columns = [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()
    return columns

def parse_feature_types(db_path, nrows=5):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM features LIMIT {nrows};"
    df_sample = pd.read_sql_query(query, conn)
    conn.close()
    columns = [c for c in df_sample.columns if c != 'sequence_id']
    feature_info = []
    for col in columns:
        col_data = df_sample[col].dropna()
        if col_data.empty:
            col_type = 'categorical'
        else:
            try:
                pd.to_numeric(col_data.astype(str))
                col_type = 'continuous'
            except ValueError:
                col_type = 'categorical'
        feature_info.append({'column_name': col, 'type': col_type})
    return feature_info

def get_available_coordinate_systems(db_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.execute("SELECT load_extension('mod_spatialite');")
    try:
        cursor = conn.execute("PRAGMA table_info(coordinates);")
        all_cols = cursor.fetchall()
        coord_systems = []
        for info in all_cols:
            name = info[1]
            if name not in ('header','id') and not name.lower().startswith('st_'):
                coord_systems.append(name)
    finally:
        conn.close()
    return coord_systems

def load_coordinates_from_db(db_path, coordinate_systems,
                             feature_col=None, filter_values=None):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.execute("SELECT load_extension('mod_spatialite');")
    try:
        #select = ["c.header"]
        select = ["c.sequence_id AS header"]
        cols = set()
        if feature_col:
            cols.add(feature_col)
        if filter_values:
            cols.update(filter_values.keys())
        for cs in coordinate_systems:
            select.append(f"ST_AsText(c.{cs}) AS {cs}_geom")
        for c in cols:
            select.append(f"f.{c} AS {c}")
        clause = ", ".join(select)
        #query = f"""
        #    SELECT {clause}
        #    FROM coordinates AS c
        #    JOIN features AS f ON c.header = f.sequence_id
        #"""
        query = f"""
            SELECT {clause}
            FROM coordinates AS c
            JOIN features AS f ON c.sequence_id = f.sequence_id
        """
        conds = []
        if filter_values:
            for col, b in filter_values.items():
                if b.get("min") is not None:
                    conds.append(f"f.{col} >= {b['min']}")
                if b.get("max") is not None:
                    conds.append(f"f.{col} <= {b['max']}")
                if b.get("cat"):
                    cats = ", ".join(f"'{v}'" for v in b["cat"])
                    conds.append(f"f.{col} IN ({cats})")
        if conds:
            query += " WHERE " + " AND ".join(conds)
        df = pd.read_sql_query(query, conn)
        for cs in coordinate_systems:
            geom = f"{cs}_geom"
            if geom in df:
                df[[f"x_{cs}", f"y_{cs}"]] = (
                    df[geom]
                      .str.extract(r'POINT\(([-\d.]+)\s+([-.\d]+)\)')
                      .astype(float)
                )
        df.drop(columns=[f"{cs}_geom" for cs in coordinate_systems],
                inplace=True, errors='ignore')
        keep = ["header"]
        for cs in coordinate_systems:
            keep += [f"x_{cs}", f"y_{cs}"]
        keep += list(cols)
        df = df[keep]
    finally:
        conn.close()
    return df

def create_datashader_image(df, x_col, y_col,
                            color_col=None, color_palette="viridis",
                            px_spread=3):
    # Resolution: change plot_width/plot_height to adjust image size
    cvs = ds.Canvas(plot_width=2000, plot_height=2000)
    if color_col and color_col in df:
        if pd.api.types.is_numeric_dtype(df[color_col]):
            agg = cvs.points(df, x_col, y_col, ds.mean(color_col))
            cmap = cm.get_cmap(color_palette) if color_palette in plt.colormaps() else cm.get_cmap("viridis")
            cmap_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
            img = tf.shade(agg, cmap=cmap_list, how='linear')
        else:
            df[color_col] = df[color_col].astype('category')
            cats = df[color_col].cat.categories
            if color_palette == 'Category10':
                palette = bokeh.palettes.Category10[10]
            elif color_palette == 'glasbey':
                palette = glasbey[:len(cats)]
            else:
                palette = [matplotlib.colors.rgb2hex(cm.get_cmap(color_palette)(i)) for i in range(256)]
            if len(cats) > len(palette):
                palette = glasbey[:len(cats)]
            key = dict(zip(cats, palette))
            agg = cvs.points(df, x_col, y_col, ds.count_cat(color_col))
            img = tf.shade(agg, color_key=key, how='eq_hist')
    else:
        agg = cvs.points(df, x_col, y_col, ds.count())
        img = tf.shade(agg, cmap=fire, how='eq_hist')

    # Make points appear larger
    if px_spread is None or px_spread <= 0:
        px_spread = 1
    img = tf.spread(img, px=int(px_spread))

    return img.to_pil()

##############################
# BUILD SIDEBAR + LAYOUT
##############################
def build_dynamic_sidebar(feature_info):
    children = [
        html.H2("Filters", style={"color":"white"}),
        html.Hr(),
        html.P("Data Filters", className="lead", style={"color":"white"})
    ]
    for feat in feature_info:
        nm, tp = feat['column_name'], feat['type']
        lbls = {"color":"white"}
        if tp == 'continuous':
            children += [
                html.Label(f"{nm} (Continuous)", style=lbls),
                html.Div([
                    dcc.Input(id={"type":"continuous-filter-min","column_name":nm},
                              type="number", placeholder=f"Min {nm}",
                              style={"width":"45%","margin-right":"5px"}),
                    dcc.Input(id={"type":"continuous-filter-max","column_name":nm},
                              type="number", placeholder=f"Max {nm}",
                              style={"width":"45%"})
                ], style={"margin-bottom":"10px"})
            ]
        else:
            children += [
                html.Label(f"{nm} (Categorical)", style=lbls),
                dcc.Checklist(
                    id={"type":"cat-checklist","column_name":nm},
                    options=[], value=[],
                    labelStyle={'display':'block','color':'white'},
                    style={"margin-bottom":"10px"}
                )
            ]
    coord_opts = get_available_coordinate_systems(db_path)
    children += [
        html.Label("Select Coordinate Systems", style={"color":"white"}),
        dcc.Checklist(
            id="coordinate-systems-checklist",
            options=[{"label":cs,"value":cs} for cs in coord_opts],
            value=[], labelStyle={'display':'block','color':'white'},
            style={"margin-bottom":"10px"}
        ),
        html.Label("Color By Feature", style={"color":"white"}),
        dcc.Dropdown(
            id="color-feature-dropdown",
            options=[{"label":f['column_name'],"value":f['column_name']} for f in feature_info],
            value=None, placeholder="Select a feature to color by",
            style={"margin-bottom":"10px"}
        ),
        html.Label("Color Palette", style={"color":"white"}),
        dcc.Dropdown(
            id="color-palette-dropdown",
            options=[
                {"label":"viridis","value":"viridis"},
                {"label":"plasma","value":"plasma"},
                {"label":"inferno","value":"inferno"},
                {"label":"Category10 (cat)","value":"Category10"},
                {"label":"glasbey (cat)","value":"glasbey"}
            ],
            value="viridis", style={"margin-bottom":"10px"}
        ),
        html.Label("Pixel spread (Datashader px)", style={"color": "white"}),
        dcc.Input(
            id="px-spread-input",
            type="number",
            min=1,
            step=1,
            value=3,
            style={"margin-bottom": "10px", "width": "100%"}
        ),
        dbc.Button("Update Plots", id="update-plots-button",
                   color="primary", className="mt-2"),
        dbc.Button("Export Bins",  id="export-bins-button",
                   color="primary", className="mt-2"),
        dbc.Button("Delete DF",    id="delete-df-button",
                   color="danger",  className="mt-2"),
    ]
    style = {
        "position":"fixed","top":0,"left":0,"bottom":0,
        "width":"20rem","padding":"2rem 1rem",
        "background-color":"#343a40","overflow-y":"auto"
    }
    return html.Div(children, style=style)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
feature_info = parse_feature_types(db_path)
sidebar = build_dynamic_sidebar(feature_info)

# Stores
store_bins    = dcc.Store(id='bins-store',    data=[])
store_overlay = dcc.Store(id='overlay-store', data=[])

app.layout = html.Div([
    sidebar,
    store_bins,
    store_overlay,
    html.Div([
        html.H3("Dynamic Coordinate System Plots", style={"color":"white"}),
        html.Div(id='coordinates-plots-container'),
        html.Div(
            id='color-legend-container',
            style={
                'color': 'white',
                'textAlign': 'center',
                'marginTop': '10px',
                'marginBottom': '20px'
            }
        ),
        html.Div([
            dcc.Input(id='bin-name-input', type='text',
                      placeholder='Enter bin name',
                      style={'margin-right':'10px'}),
            html.Button('Create Bin',  id='create-bin-button',  n_clicks=0),
            html.Button('Inspect Bin', id='inspect-bin-button', n_clicks=0,
                        style={'margin-left':'10px'}),
            html.Button('Overlay Points', id='overlay-points-button',
                        n_clicks=0, style={'margin-left':'10px'}),
        ], style={'margin-bottom':'20px'}),
        html.H4("Bins List", style={"color":"white"}),
        html.Div(id='bin-list-container',
                 style={"color":"white","margin-bottom":"20px"}),
        html.Div(id='error-message',
                 style={'color':'red','margin-bottom':'10px'}),
        html.Div(id='bin-table-container',
                 style={'margin-top':'20px'}),
    ], style={'margin-left':'22rem','padding':'2rem'})
], style={'backgroundColor':'#2b2b2b'})


##############################
# CATEGORICAL OPTIONS CALLBACK
##############################
@app.callback(
    Output({'type':'cat-checklist','column_name':MATCH}, 'options'),
    Input({'type':'cat-checklist','column_name':MATCH}, 'id')
)
def populate_categorical_options(matching_id):
    col = matching_id["column_name"]
    conn = sqlite3.connect(db_path)
    try:
        res = conn.execute(f"SELECT DISTINCT {col} FROM features "
                           f"ORDER BY {col} LIMIT 200;").fetchall()
        unique_vals = [r[0] for r in res if r[0] is not None]
    finally:
        conn.close()
    return [{"label":str(v),"value":v} for v in unique_vals]


##############################
# MAIN PLOT CALLBACK (RESPONSIVE SIZING)
##############################
@app.callback(
    Output('coordinates-plots-container','children'),
    Output('color-legend-container','children'),
    Input('update-plots-button','n_clicks'),
    Input('overlay-store','data'),
    State('coordinate-systems-checklist','value'),
    State('color-feature-dropdown','value'),
    State('color-palette-dropdown','value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'id'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'id'),
    State({'type':'cat-checklist','column_name':ALL}, 'value'),
    State({'type':'cat-checklist','column_name':ALL}, 'id'),
    State('px-spread-input', 'value'),
    prevent_initial_call=True
)
def update_multiple_coord_plots(
    _btn, overlay_headers,
    selected_coords, selected_feature, selected_palette,
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids,
    px_spread
):
    if not selected_coords:
        return [], ""

    # build filter_values
    fv = {}
    for v, i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["min"] = v
    for v, i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["max"] = v
    for vals, i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["cat"] = vals

    df = load_coordinates_from_db(
        db_path=db_path,
        coordinate_systems=selected_coords,
        feature_col=selected_feature,
        filter_values=fv
    )

    # Dynamically decide layout based on number of plots
    n_plots = len(selected_coords)
    if n_plots == 1:
        plots_per_row = 1
        graph_height = 800
    elif n_plots == 2:
        plots_per_row = 2
        graph_height = 600
    elif n_plots == 3:
        plots_per_row = 3
        graph_height = 500
    else:
        plots_per_row = 3
        graph_height = 400

    rows = ceil(n_plots / plots_per_row)
    row_comps = []

    for r in range(rows):
        slice_cs = selected_coords[r*plots_per_row:(r+1)*plots_per_row]
        cols = []
        row_count = len(slice_cs)
        col_width = max(1, int(12 / row_count))  # Bootstrap col widths

        for cs in slice_cs:
            xcol, ycol = f"x_{cs}", f"y_{cs}"

            fig = go.Figure()

            # always add the original raster below
            base_img = create_datashader_image(
                df, xcol, ycol, selected_feature, selected_palette,
                px_spread=px_spread
            )
            fig.add_layout_image({
                "source": base_img,
                "xref": "x", "yref": "y",
                "x": df[xcol].min(), "y": df[ycol].max(),
                "sizex": df[xcol].max() - df[xcol].min(),
                "sizey": df[ycol].max() - df[ycol].min(),
                "sizing": "stretch",
                "opacity": 1.0,
                "layer": "below"
            })

            # if overlaying, add red/green map above at 50% opacity
            if overlay_headers:
                df['overlay_flag'] = df['header'].isin(overlay_headers).astype('category')
                cvs = ds.Canvas(plot_width=500, plot_height=500)
                agg = cvs.points(df, xcol, ycol, ds.count_cat('overlay_flag'))
                color_key = {False: 'red', True: 'green'}
                overlay_img = tf.shade(agg, color_key=color_key, how='eq_hist')

                if px_spread is None or px_spread <= 0:
                    px_spread_use = 1
                else:
                    px_spread_use = int(px_spread)

                overlay_img = tf.spread(overlay_img, px=px_spread_use)
                overlay_img = overlay_img.to_pil().convert("RGBA")
                fig.add_layout_image({
                    "source": overlay_img,
                    "xref": "x", "yref": "y",
                    "x": df[xcol].min(), "y": df[ycol].max(),
                    "sizex": df[xcol].max() - df[xcol].min(),
                    "sizey": df[ycol].max() - df[ycol].min(),
                    "sizing": "stretch",
                    "opacity": 1.0,
                    "layer": "above"
                })

            fig.update_layout(
                title=f"{cs} Plot",
                dragmode='lasso',
                height=graph_height,
                xaxis=dict(visible=False,
                           range=[df[xcol].min(), df[xcol].max()]),
                yaxis=dict(visible=False,
                           range=[df[ycol].min(), df[ycol].max()]),
                margin=dict(l=0, r=0, t=30, b=0)
            )

            # Wrap graph in a resizable container
            graph_container = html.Div(
                dcc.Graph(
                    id={"type":"scatter-plot","index":cs},
                    figure=fig,
                    config={"responsive": True},
                    style={"height": "100%", "width": "100%"}
                ),
                style={
                    "height": f"{graph_height}px",
                    "resize": "both",
                    "overflow": "auto",
                    "border": "1px solid #444"
                }
            )

            cols.append(
                dbc.Col([
                    html.H5(cs, style={"color":"white","textAlign":"center"}),
                    graph_container
                ], width=col_width)
            )

        row_comps.append(dbc.Row(cols, justify="center", className="mb-3"))

    # ---------- build shared legend below grid ----------
    legend_component = ""

    if selected_feature and (selected_feature in df.columns):
        # Continuous feature -> colorbar
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            if df[selected_feature].notna().any():
                vmin = float(df[selected_feature].min())
                vmax = float(df[selected_feature].max())

                if vmin != vmax:
                    cmap_name = selected_palette if selected_palette in plt.colormaps() else "viridis"
                    cmap = cm.get_cmap(cmap_name)

                    colorscale = []
                    for i in range(256):
                        frac = i / 255.0
                        r, g, b, _ = cmap(frac)
                        colorscale.append([
                            frac,
                            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                        ])

                    legend_trace = go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='markers',
                        marker=dict(
                            size=0.0001,  # effectively invisible
                            color=[vmin, vmax],
                            colorscale=colorscale,
                            showscale=True,
                            colorbar=dict(
                                title=dict(text=selected_feature, font=dict(color='white')),
                                tickfont=dict(color='white'),
                                orientation='h',
                                x=0.5,
                                xanchor='center',
                                thickness=15,
                                len=0.8,
                            ),
                        ),
                        hoverinfo='none',
                        showlegend=False
                    )
                    
                    legend_fig = go.Figure(data=[legend_trace])
                    legend_fig.update_xaxes(visible=False)
                    legend_fig.update_yaxes(visible=False)
                    legend_fig.update_layout(
                        margin=dict(l=40, r=40, t=10, b=20),
                        height=120,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    legend_component = dcc.Graph(
                        figure=legend_fig,
                        style={"height": "120px"}
                    )
                else:
                    legend_component = html.Div(
                        f"Legend: {selected_feature} has a constant value ({vmin}).",
                        style={"color": "white"}
                    )

        # Categorical feature -> HTML legend
        else:
            cats = df[selected_feature].astype("category").cat.categories.tolist()
            if cats:
                # Reproduce palette logic from create_datashader_image
                if selected_palette == "Category10":
                    palette = bokeh.palettes.Category10[10]
                elif selected_palette == "glasbey":
                    palette = glasbey[:len(cats)]
                else:
                    base_cmap_name = selected_palette if selected_palette in plt.colormaps() else "viridis"
                    base_cmap = cm.get_cmap(base_cmap_name)
                    palette = []
                    n = max(len(cats) - 1, 1)
                    for i in range(len(cats)):
                        frac = i / n if n > 0 else 0.0
                        r, g, b, _ = base_cmap(frac)
                        palette.append(matplotlib.colors.rgb2hex((r, g, b)))

                if len(cats) > len(palette):
                    palette = glasbey[:len(cats)]

                items = []
                for cat, color in zip(cats, palette):
                    items.append(
                        html.Div(
                            [
                                html.Span(
                                    style={
                                        "display": "inline-block",
                                        "width": "12px",
                                        "height": "12px",
                                        "marginRight": "6px",
                                        "backgroundColor": color,
                                        "border": "1px solid #ccc",
                                        "verticalAlign": "middle"
                                    }
                                ),
                                html.Span(str(cat), style={"verticalAlign": "middle"})
                            ],
                            style={
                                "display": "inline-block",
                                "marginRight": "12px",
                                "marginBottom": "4px"
                            }
                        )
                    )

                legend_component = html.Div(
                    [
                        html.Div(
                            f"Legend: {selected_feature}",
                            style={"marginBottom": "4px"}
                        ),
                        html.Div(items)
                    ],
                    style={"color": "white"}
                )

    # No feature selected -> textual note for default fire density map
    else:
        legend_component = html.Div(
            "Color map: point density (Datashader 'fire' colormap, eq_hist). "
            "Brighter = higher local density.",
            style={"color": "white"}
        )

    return row_comps, legend_component


##############################
# OVERLAY POINTS CALLBACK
##############################
@app.callback(
    Output('overlay-store','data'),
    Input('overlay-points-button','n_clicks'),
    State({'type':'scatter-plot','index':ALL}, 'selectedData'),
    State('coordinate-systems-checklist','value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-min','column_name':ALL}, 'id'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'value'),
    State({'type':'continuous-filter-max','column_name':ALL}, 'id'),
    State({'type':'cat-checklist','column_name':ALL}, 'value'),
    State({'type':'cat-checklist','column_name':ALL}, 'id'),
    prevent_initial_call=True
)
def overlay_points(
    n, all_sel, selected_coords,
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids
):
    fv = {}
    for v,i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["min"]=v
    for v,i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["max"]=v
    for vals,i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["cat"]=vals

    df = load_coordinates_from_db(
        db_path=db_path,
        coordinate_systems=selected_coords,
        feature_col=None,
        filter_values=fv
    )
    headers = set()
    for sel, cs in zip(all_sel, selected_coords):
        if sel and 'lassoPoints' in sel:
            xcol, ycol = f"x_{cs}", f"y_{cs}"
            pts = list(zip(sel['lassoPoints']['x'], sel['lassoPoints']['y']))
            path = Path(pts)
            mask = path.contains_points(df[[xcol,ycol]].values)
            headers.update(df.loc[mask, 'header'])
    return list(headers)


##############################
# BINS + INSPECT + EXPORT (FASTQ/FASTA, CHUNKED, FULL HEADER)
##############################
@app.callback(
    [
        Output('bins-store','data'),
        Output('error-message','children'),
        Output('bin-table-container','children'),
        Output('bin-list-container','children')
    ],
    [
        Input('create-bin-button','n_clicks'),
        Input('inspect-bin-button','n_clicks'),
        Input('export-bins-button','n_clicks'),
    ],
    [
        State({'type':'scatter-plot','index':ALL}, 'selectedData'),
        State('bin-name-input','value'),
        State('bins-store','data'),
        State({'type':'continuous-filter-min','column_name':ALL}, 'value'),
        State({'type':'continuous-filter-min','column_name':ALL}, 'id'),
        State({'type':'continuous-filter-max','column_name':ALL}, 'value'),
        State({'type':'continuous-filter-max','column_name':ALL}, 'id'),
        State({'type':'cat-checklist','column_name':ALL}, 'value'),
        State({'type':'cat-checklist','column_name':ALL}, 'id'),
        State('color-feature-dropdown','value'),
        State('coordinate-systems-checklist','value'),
    ],
    prevent_initial_call=True
)
def handle_bin_operations(
    create_clicks, inspect_clicks, export_clicks,
    all_sel, bin_name, bins_data,
    all_min_vals, all_min_ids,
    all_max_vals, all_max_ids,
    all_cat_vals, all_cat_ids,
    color_feature, selected_coords
):
    ctx_trig = ctx.triggered_id
    error_message = ""
    table_content = None
    if not bins_data:
        bins_data = []

    fv = {}
    for v,i in zip(all_min_vals, all_min_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["min"]=v
    for v,i in zip(all_max_vals, all_max_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["max"]=v
    for vals,i in zip(all_cat_vals, all_cat_ids):
        fv.setdefault(i["column_name"],{"min":None,"max":None,"cat":[]} )["cat"]=vals

    if ctx_trig == 'create-bin-button':
        if not any(all_sel) or not bin_name or bin_name.strip()=="":
            error_message = "Please select points and provide a bin name."
        else:
            polys = {
                cs: list(zip(sel['lassoPoints']['x'], sel['lassoPoints']['y']))
                for sel,cs in zip(all_sel, selected_coords)
                if sel and 'lassoPoints' in sel
            }
            bins_data.append({
                "bin_name": bin_name,
                "filters": fv,
                "coordinate_systems": selected_coords,
                "polygons": polys
            })
            error_message = f"Bin '{bin_name}' created successfully."

    elif ctx_trig == 'inspect-bin-button':
        if not any(all_sel):
            error_message = "No lasso selection to inspect."
        else:
            df_master = load_coordinates_from_db(
                db_path=db_path,
                coordinate_systems=selected_coords,
                feature_col=color_feature,
                filter_values=fv
            )
            idxs = set()
            for sel, cs in zip(all_sel, selected_coords):
                if sel and 'lassoPoints' in sel:
                    xcol, ycol = f"x_{cs}", f"y_{cs}"
                    pts = list(zip(sel['lassoPoints']['x'], sel['lassoPoints']['y']))
                    path = Path(pts)
                    mask = path.contains_points(df_master[[xcol,ycol]].values)
                    idxs.update(df_master.index[mask])
            if idxs:
                sub_df = df_master.loc[list(idxs)].copy()
                table_content = dash_table.DataTable(
                    columns=[{"name":c,"id":c} for c in sub_df.columns],
                    data=sub_df.to_dict("records"),
                    page_size=10,
                    style_table={"overflowX":"auto"},
                    style_header={"backgroundColor":"#343a40","color":"white"},
                    style_data={"backgroundColor":"#2b2b2b","color":"white"},
                )
                error_message = f"Inspect Bin: {len(sub_df)} points found."
            else:
                error_message = "Inspect Bin: 0 points found."

    elif ctx_trig == 'export-bins-button':
        if not bins_data:
            error_message = "No bins to export."
        else:
            os.makedirs(output_dir_default, exist_ok=True)
            for bin_entry in bins_data:
                df_bin = load_coordinates_from_db(
                    db_path=db_path,
                    coordinate_systems=bin_entry["coordinate_systems"],
                    feature_col=None,
                    filter_values=bin_entry["filters"]
                )
                idxs = set()
                for cs,poly in bin_entry["polygons"].items():
                    xcol, ycol = f"x_{cs}", f"y_{cs}"
                    path = Path(poly)
                    mask = path.contains_points(df_bin[[xcol,ycol]].values)
                    idxs.update(df_bin.index[mask])
                df_filt = df_bin.loc[list(idxs)]

                # Export CSV as before
                csv_path = os.path.join(
                    output_dir_default,
                    f"{bin_entry['bin_name']}.csv"
                )
                df_filt.to_csv(csv_path, index=False)

                # Chunked FASTQ/FASTA export to avoid "too many SQL variables"
                if 'header' in df_filt:
                    headers = df_filt['header'].unique().tolist()
                    if headers:
                        conn = sqlite3.connect(db_path)
                        conn.enable_load_extension(True)
                        conn.execute("SELECT load_extension('mod_spatialite');")
                        try:
                            cursor = conn.cursor()
                            cursor.execute("PRAGMA table_info(fasta);")
                            cols = [r[1] for r in cursor.fetchall()]
                            has_qualities_col = 'qualities' in cols
                            has_full_header_col = 'full_header' in cols

                            select_cols_list = ["header", "sequence"]
                            if has_full_header_col:
                                select_cols_list.insert(1, "full_header")
                            if has_qualities_col:
                                select_cols_list.append("qualities")
                            select_cols = ", ".join(select_cols_list)

                            # Chunk headers to avoid SQLite var limit
                            chunk_size = 900  # safely under default 999 limit
                            dfs = []
                            for i in range(0, len(headers), chunk_size):
                                batch = headers[i:i+chunk_size]
                                placeholders = ", ".join("?" * len(batch))
                                fasta_q = (
                                    f"SELECT {select_cols} FROM fasta "
                                    f"WHERE header IN ({placeholders});"
                                )
                                df_chunk = pd.read_sql_query(
                                    fasta_q, conn, params=batch
                                )
                                if not df_chunk.empty:
                                    dfs.append(df_chunk)

                            if not dfs:
                                continue  # nothing to write for this bin

                            fasta_df = pd.concat(dfs, ignore_index=True)

                            # Decide FASTQ vs FASTA
                            write_fastq = (
                                has_qualities_col
                                and 'qualities' in fasta_df.columns
                                and fasta_df['qualities'].notna().any()
                            )

                            def get_output_header(row):
                                if has_full_header_col and 'full_header' in row and not pd.isna(row['full_header']):
                                    return row['full_header']
                                return row['header']

                            if write_fastq:
                                out_path = os.path.join(
                                    output_dir_default,
                                    f"{bin_entry['bin_name']}.fastq"
                                )
                                with open(out_path, 'w') as fh:
                                    for _, row in fasta_df.iterrows():
                                        q = row.get('qualities')
                                        header_out = get_output_header(row)
                                        if pd.isna(q):
                                            # Fallback: write that read as FASTA if no qual
                                            fh.write(
                                                f">{header_out}\n{row['sequence']}\n"
                                            )
                                        else:
                                            fh.write(
                                                f"@{header_out}\n"
                                                f"{row['sequence']}\n"
                                                f"+\n"
                                                f"{q}\n"
                                            )
                            else:
                                out_path = os.path.join(
                                    output_dir_default,
                                    f"{bin_entry['bin_name']}.fasta"
                                )
                                with open(out_path, 'w') as fh:
                                    for _, row in fasta_df.iterrows():
                                        header_out = get_output_header(row)
                                        fh.write(
                                            f">{header_out}\n{row['sequence']}\n"
                                        )
                        finally:
                            conn.close()

            error_message = (
                f"Bins exported to {output_dir_default} "
                f"(FASTQ if qualities present, otherwise FASTA; using full headers when available)."
            )

    bin_list_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Bin Name"), html.Th("Coordinate Systems"),
            html.Th("Polygon(s)"), html.Th("Filters")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(b["bin_name"]),
                html.Td(", ".join(b["coordinate_systems"])),
                html.Td(str(b["polygons"])),
                html.Td(str(b["filters"]))
            ]) for b in bins_data
        ])
    ], style={"color":"white","border":"1px solid #fff"})

    return bins_data, error_message, table_content, bin_list_table

if __name__ == '__main__':
    app.run(debug=False)
