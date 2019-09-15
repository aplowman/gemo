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
    'fill_colour_max': {
        'ensure_exists': {'marker': {'cmax': None}},
        'set': ['marker', 'cmax'],
    },
    'fill_colour_min': {
        'ensure_exists': {'marker': {'cmin': None}},
        'set': ['marker', 'cmin'],
    },
    'fill_colourscale': {
        'ensure_exists': {'marker': {'colorscale': None}},
        'set': ['marker', 'colorscale'],
    },
    'outline_colour': {
        'ensure_exists': {'marker': {'line': {'color': None}}},
        'set': ['marker', 'line', 'color'],
    },
    'outline_size': {
        'ensure_exists': {'marker': {'line': {'width': None}}},
        'set': ['marker', 'line', 'width'],
    },
    'marker_size': {
        'ensure_exists': {'marker': {'size': None}},
        'set': ['marker', 'size'],
    },
    'marker_symbol': {
        'ensure_exists': {'marker': {'symbol': None}},
        'set': ['marker', 'symbol'],
    },
}


def extract_styles(trace_dict):
    'Extract styles and return in a format suitable for Plotly.'

    if 'styles' not in trace_dict:
        return {}

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


def make_figure(data, layout_args, dimension):

    scatter_type = 'scatter3d' if dimension == 3 else 'scattergl'

    plot_data = []

    for i in data['boxes']:
        plot_data.append({
            **i,
            'type': scatter_type,
            'mode': 'lines',
        })

    for i in data['lines']:
        plot_data.append({
            **i,
            'type': scatter_type,
            'mode': 'lines',
        })

    for i in data['points']:
        styles = extract_styles(i)
        plot_data.append({
            **i,
            **styles,
            'type': scatter_type,
            'mode': 'markers',
        })

    if layout_args is None:
        layout_args = {}

    plotly_args = {
        'xaxis': layout_args.get('xaxis', {}),
        'yaxis': layout_args.get('yaxis', {}),
        'width': layout_args.get('width', None),
        'height': layout_args.get('height', None),
    }

    if dimension == 2:
        plotly_args['xaxis'].update({
            'scaleanchor': 'y',
        })

    elif dimension == 3:
        plotly_args.update({
            'scene': {
                'aspectmode': 'data',
                'camera': {
                    'projection': {
                        'type': 'orthographic',
                    }
                }
            }
        })

    fig = go.FigureWidget(data=plot_data, layout=plotly_args)
    return fig
