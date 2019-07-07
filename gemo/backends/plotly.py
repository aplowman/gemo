'`geom.backends.plotly.py'

from plotly import graph_objs as go


def make_figure(data, layout_args):

    plot_data = []
    for i in data['points']:
        plot_data.append({
            **i,
            'type': 'scatter3d',
            'mode': 'markers',
        })

    for i in data['boxes']:
        plot_data.append({
            **i,
            'type': 'scatter3d',
            'mode': 'lines',
        })

    if layout_args is None:
        layout_args = {}

    layout_args.update({
        'plotly_args': {
            'scene': {
                'aspectmode': 'data',
                'camera': {
                    'projection': {
                        'type': 'orthographic',
                    }
                }
            }
        }
    })

    fig = go.FigureWidget(data=plot_data, layout=layout_args['plotly_args'])
    return fig
