from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go



fig = go.Figure()

df = pd.read_csv("plot/plot_B32_T1024_I512", index_col=0)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['times'],
    hovertext=df['times'],
    hoverinfo="text",
    name='plot_B32_T1024_I512',
    mode='markers',
    marker=dict(
        color="blue"
    ) ,
    showlegend=True
))

# add line / trace 2 to figure
df = pd.read_csv("plot/plot_B32_T1024_I1024", index_col=0)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['times'],
    hovertext=df['times'],
    hoverinfo="text",
    name='plot_B32_T1024_I1024',
    mode='markers',
    marker=dict(
        color="green"
    ),
    showlegend=True
))
# add line
df = pd.read_csv("plot/plot_B16_T1024_I1024", index_col=0)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['times'],
    hovertext=df['times'],
    hoverinfo="text",
    name='plot_B16_T1024_I1024',
    mode='markers',
    marker=dict(
        color="red"
    ),
    showlegend=True
))
# add line
df = pd.read_csv("plot/plot_B64_T1024_I1024", index_col=0)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['times'],
    hovertext=df['times'],
    hoverinfo="text",
    name='plot_B64_T1024_I1024',
    mode='markers',
    marker=dict(
        color="yellow"
    ),
    showlegend=True
))

#######
fig.write_html("plot.html")
fig.show()


