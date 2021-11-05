import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plot([go.Scatter(x=[1, 2, 3], y=[3, 1, 6])])