'`geom.backends.plotly.py'

from plotly import graph_objs as go


def make_figure(geom_group, layout_args, group_points_by):

    plot_data = []

    if geom_group.boxes is not None:
        for box_label, box in geom_group.boxes.items():
            box_xyz = box.corner_coords
            plot_data.append({
                'type': 'scatter3d',
                'x': box_xyz[0],
                'y': box_xyz[1],
                'z': box_xyz[2],
                'mode': 'lines',
                'name': box_label,
            })

    if geom_group.points is not None:
        for point_label, point_set in geom_group.points.items():
            plot_data.append({
                **point_set.get_plot_data(),
                'name': point_label,
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
