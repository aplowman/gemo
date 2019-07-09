'`geom.backends.plotly.py'

from pprint import pprint
import copy

from plotly import graph_objs as go

from gemo.utils import update_dict, set_in_dict


PLOTLY_STYLES = {
    'fill_colour': {
        'ensure_exists': {'marker': {'color': None}},
        'set': ['marker', 'color'],
    },
    'outline_colour': {
        'ensure_exists': {'marker': {'line': {'color': None, 'width': 3}}},
        'set': ['marker', 'line', 'color'],
    }
}


def extract_styles(trace_dict):
    'Extract styles and return in a format suitable for Plotly.'

    styles = trace_dict.pop('styles')

    if not set(styles.keys()).issubset(PLOTLY_STYLES.keys()):
        msg = ('Style named "{}" is not supported by the Plotly backend.')
        raise NotImplementedError(msg)

    new_styles = copy.deepcopy(styles)
    for style_name in styles.keys():
        style_spec = PLOTLY_STYLES[style_name]
        update_dict(new_styles, style_spec['ensure_exists'])
        set_in_dict(new_styles, style_spec['set'], styles[style_name])
        new_styles.pop(style_name)

    return new_styles


def make_figure(data, layout_args):

    plot_data = []
    for i in data['points']:
        styles = extract_styles(i)
        plot_data.append({
            **i,
            **styles,
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
