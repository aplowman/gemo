'`gemo.gemo.py`'

import numpy as np
from vecmaths import geometry
from spatial_sites import Sites
from spatial_sites.utils import repr_dict
from pprint import pprint

from gemo.geometries import Box
from gemo.backends import make_figure_func
from gemo.utils import nest, validate_3d_vector


class ViewOrientation(object):
    """
    Attributes
    ----------
    look_at : ndarray of shape (3, 1)
        Camera direction.
    up : ndarray of shape (3, 1)
        Up direction.
    """

    def __init__(self, look_at, up):

        look_at = validate_3d_vector(look_at)
        up = validate_3d_vector(up)

        # Normalise:
        self.look_at = (look_at / np.linalg.norm(look_at))
        self.up = (up / np.linalg.norm(up))

        if np.all(np.isclose(self.look_at, self.up)):
            raise ValueError('`look_at` and `up` cannot be the same!')

    def __repr__(self):

        out = '{}(look_at={!r}, up={!r})'.format(
            self.__class__.__name__,
            self.look_at,
            self.up
        )
        return out

    @property
    def rotation_matrix(self):
        """Find the rotation matrix to rotate the supercell and atoms,
        such that `up` becomes the y-axis and `eye` becomes the negative
        of the projection plane normal.

        """

        u_x = np.cross(self.look_at, self.up, axis=0)
        u_x = u_x / np.linalg.norm(u_x)
        u_y = np.cross(u_x, self.look_at, axis=0)
        u_z = -self.look_at

        rot_mat = np.vstack([u_x.T, u_y.T, u_z.T])
        return rot_mat


class ViewFrustum(object):

    def __init__(self, left, right, top, bottom, near, far):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.far = far

        self.box = Box(
            edge_vectors=np.array([
                [right - left, 0, 0],
                [0, top - bottom, 0],
                [0, 0, far - near],
            ]).T
        )

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.top - self.bottom

    @property
    def depth(self):
        return self.far - self.near

    @property
    def projection_matrix(self):
        """The projection matrix is a scaling followed by a translation. It
        works to map coordinates within the viewing frustum"""

        proj = np.array([
            [2/self.width, 0, 0, -(self.right + self.left)/self.width],
            [0, 2/self.height, 0, -(self.top + self.bottom)/self.height],
            [0, 0, -2/self.depth, -(self.far + self.near)/self.depth],
            [0, 0, 0, 1],
        ])
        return proj


class GeometryGroup(object):

    def __init__(self, points=None, boxes=None):

        self.points = self._validate_points(points)
        self.boxes = self._validate_boxes(boxes)

    def __repr__(self):

        points = repr_dict(self.points, indent=4)
        boxes = repr_dict(self.boxes, indent=4)

        indent = ' ' * 4
        out = (
            '{0}(\n'
            '{1}points={2},\n'
            '{1}boxes={3},\n'
            ')'.format(
                self.__class__.__name__,
                indent,
                points,
                boxes,
            )
        )
        return out

    def __copy__(self):
        points = copy.deepcopy(self.points)
        boxes = copy.deepcopy(self.boxes)
        return GeometryGroup(points=points, boxes=boxes)

    def _validate_points(self, points):

        if not points:
            return {}

        msg = ('`points` must be a dict whose keys are strings and whose '
               'values are `Sites` objects.')
        if not isinstance(points, dict):
            raise ValueError(msg)

        for val in points.values():
            if not isinstance(val, Sites):
                raise ValueError(msg)

        return points

    def _validate_boxes(self, boxes):

        if not boxes:
            return {}

        msg = ('`boxes` must be a dict whose keys are strings and whose '
               'values are `Box` objects.')
        if not isinstance(boxes, dict):
            raise ValueError(msg)

        for val in boxes.values():
            if not isinstance(val, Box):
                raise ValueError(msg)

        return boxes

    def _validate_points_grouping(self, group_dict):

        ALLOWED_STYLES = [
            'fill_colour',
            'outline_colour',
            'test',
        ]

        if not group_dict:
            return {}

        for points_name, points_grouping in group_dict.items():
            if points_name not in self.points:
                msg = 'No points with name "{}" exist.'
                raise ValueError(msg.format(points_name))

            if not isinstance(points_grouping, list):
                msg = ('For points named "{}", specify the grouping labels as '
                       'a list of dicts.')
                raise ValueError(msg.format(points_name))

            # Not allowed to repeat the same style for different labels:
            used_styles = []

            for i in points_grouping:
                if 'label' not in i:
                    msg = ('Must supply a points label by which to group the '
                           'coordinates.')
                    raise ValueError(msg)

                if i['label'] not in self.points[points_name].labels:
                    msg = ('Cannot find points label named "{}" for the "{}" '
                           'points.')
                    raise ValueError(msg.format(i['label'], points_name))

                if 'styles' in i:

                    label = self.points[points_name].labels[i['label']]
                    unique_vals = label.unique_values

                    for style_name, style_dict in i['styles'].items():

                        if style_name not in ALLOWED_STYLES:
                            msg = ('"{}" is not an allowed styles. Allowed '
                                   'styles are: {}')
                            raise ValueError(msg.format(
                                style_name, ALLOWED_STYLES))

                        if style_name in used_styles:
                            msg = ('Style "{}" is used for multiple labels '
                                   'for points named "{}".')
                            raise ValueError(msg.format(
                                style_name, points_name))
                        else:
                            used_styles.append(style_name)

                        # check a value specified for each unique value:
                        if set(style_dict.keys()) != set(unique_vals):
                            msg = ('Specify style "{}" for each unique '
                                   '"{}" value of the points named "{}". The '
                                   'unique values are: {}.')
                            raise ValueError(
                                msg.format(style_name, i['label'], points_name,
                                           unique_vals)
                            )

        return group_dict

    def _get_plot_data(self, group_points):

        points = []
        for points_name, points_set in self.points.items():

            if points_name in group_points:
                # Split points into groups:

                group_names = [i['label'] for i in group_points[points_name]]
                uniq = [points_set.labels[i].unique_values
                        for i in group_names]

                group_name_fmt = '{}['.format(points_name)
                for i in group_names:
                    group_name_fmt += '{}: {{}}; '.format(i)
                group_name_fmt += ']'

                all_styles = {i['label']: i.get('styles', {})
                              for i in group_points[points_name]}

                for i in nest(*uniq):
                    labels_match = dict(zip(group_names, i))
                    # Maybe a Sites.subset method would be useful here to get a
                    # new Sites object with a subset of coords:
                    pts = points_set.whose(**labels_match)
                    points.append({
                        'name': group_name_fmt.format(*i),
                        'x': pts[0],
                        'y': pts[1],
                        'z': pts[2],
                        'styles': {
                            style_name: style_vals[
                                i[group_names.index(label_name)]
                            ]
                            for label_name, label_styles in all_styles.items()
                            for style_name, style_vals in label_styles.items()
                        }
                    })

        boxes = []
        for box_name, box in self.boxes.items():
            coords = box.corner_coords
            boxes.append({
                'name': box_name,
                'x': coords[0],
                'y': coords[1],
                'z': coords[2],
            })

        data = {
            'points': points,
            'boxes': boxes,
        }

        return data

    def copy(self):
        return self.__copy__()

    def show(self, group_points=None, group_boxes=None, layout_args=None,
             target='interactive', backend='plotly'):

        group_points = self._validate_points_grouping(group_points)
        plot_data = self._get_plot_data(group_points)
        fig = make_figure_func[backend](plot_data, layout_args)

        return fig

    def project(self, view_orientation, left=None, right=None, bottom=None,
                top=None, near=None, far=None, label=None):

        rot_mat = view_orientation.rotation_matrix

        # Rotate coordinates
        points_rot = []
        for point_set in self.points:
            p_rot = np.dot(rot_mat, point_set)
            points_rot.append(p_rot)

        # print('points_rot[0]: \n{}\n'.format(points_rot[0]))

        boxes_rot = []
        for box in self.boxes:
            b_rot = np.dot(rot_mat, box)
            boxes_rot.append(b_rot)

        # Get boxes vertices
        boxes_verts = []
        for box in boxes_rot:
            b_verts = geometry.get_box_xyz(box)[0]
            boxes_verts.append(b_verts)

        # Get global bounding box:
        all_coords = np.concatenate(points_rot + boxes_verts, axis=1)
        minimums = np.min(all_coords, axis=1)
        maximums = np.max(all_coords, axis=1)

        print('minimums: {}'.format(minimums))
        print('maximums: {}'.format(maximums))

        # (Later: could do above with a GeometryGroup.rotate method which generates a copy)

        # Set viewing frustum planes to bounding box if not specified
        if not left:
            left = minimums[0]
        if not right:
            right = maximums[0]
        if not bottom:
            bottom = minimums[1]
        if not top:
            top = maximums[1]
        if not near:
            near = -maximums[2]
        if not far:
            far = -minimums[2]

        # Get projection matrix
        view_frustum = ViewFrustum(left, right, top, bottom, near, far)
        proj_mat = view_frustum.projection_matrix

        # Transform to homogeneous coordinates and project
        points_proj = []
        points_inview = []
        for point_set in points_rot:

            p_homo = np.vstack([
                point_set, np.ones((1, point_set.shape[1]))
            ])
            p_proj = np.dot(proj_mat, p_homo)

            inside_idx = np.where(
                np.all(
                    np.logical_and(
                        p_proj <= 1,
                        p_proj >= -1
                    ),
                    axis=0
                )
            )[0]
            points_inview.append(inside_idx)
            points_proj.append(p_proj)

        points = {
            'rotated': points_rot,
            'projected': points_proj,
            'inview_idx': points_inview,
        }

        boxes_proj = []
        for box in boxes_verts:

            b_homo = np.vstack([
                box, np.ones((1, box.shape[1]))
            ])
            # print('b_homo: \n{}\n'.format(b_homo))
            b_proj = np.dot(proj_mat, b_homo)
            boxes_proj.append(b_proj)

        boxes = {
            'rotated': boxes_verts,
            'projected': boxes_proj,
        }

        ggp = GeometryGroupProjection(
            view_frustum, points=points, point_labels=self.point_labels, boxes=boxes,
            label=label, bounding={'min': minimums, 'max': maximums}
        )

        return ggp

    @property
    def bounding_coordinates(self):
        'Get the orthogonal bounding box minima and maxima in each dimension.'

        # Concatenate all points and box coordinates:
        points = np.hstack([i._coords for i in self.points.values()])
        box_coords = np.hstack([i.corner_coords for i in self.boxes.values()])
        all_coords = np.hstack([points, box_coords])

        out = np.array([
            np.min(all_coords, axis=1),
            np.max(all_coords, axis=1)
        ])

        return out


class GeometryGroupProjection(object):

    def __init__(self, view_frustum, points=None, boxes=None, vectors=None,
                 label=None, bounding=None, point_labels=None):
        """
        Parameters
        ----------
        points : dict
            Keys:
                rotated
                projected
                inview_idx
        boxes : dict
            Keys:
                rotated
                projected        
        """
        self.view_frustum = view_frustum
        self.points = points
        self.point_labels = point_labels or {}
        self.boxes = boxes
        self.vectors = vectors
        self.label = label
        self.bounding = bounding

    def show(self, **kwargs):

        vis = self.prepare_visual(**kwargs)
        return NotImplementedError
        # make_my_fig(**vis)

    def prepare_visual(self, plot_height=None, plot_width=None,
                       pixels_per_unit=None, target='interactive', inview=True,
                       world_coordinates=True, layout_args=None):
        """
        inview : bool
            If True, only points in the viewing frustum are plotted.
        world_coordinates : bool
            If True, plot in world coordinates (rather than camera coordinates,
            for which the boundaries of the viewing frustum are at +/-1.)
        """

        if layout_args is None:
            layout_args = {}

        plot_data = []
        if self.boxes is not None:

            if world_coordinates:
                boxes_to_show = self.boxes['rotated']
            else:
                boxes_to_show = self.boxes['projected']

            for box_idx, box in enumerate(boxes_to_show):

                if world_coordinates:
                    # Subtract bounding box mins:
                    box = box - self.bounding['min'][:, None]

                ln_width = 0.8
                ln_col = 'black'
                if box_idx < len(boxes_to_show) - 2:
                    ln_width = 0.5
                    ln_col = 'gray'

                plot_data.append({
                    'type': 'scatter',
                    'cliponaxis': False,
                    'x': box[0],
                    'y': box[1],
                    'xaxis': 'x',
                    'yaxis': 'y',
                    'mode': 'lines',
                    'line': {
                        'color': ln_col,
                        'width': ln_width,
                    }
                })

        if self.points is not None:

            if world_coordinates:
                points_to_show = self.points['rotated']
            else:
                points_to_show = self.points['projected']

            for point_set_idx, (point_proj, inview_idx) in enumerate(
                    zip(points_to_show, self.points['inview_idx'])):

                if inview:
                    point_proj = point_proj[:, inview_idx]
                    if point_set_idx == len(points_to_show) - 1:
                        for lab_name, lab_val in self.point_labels.items():
                            self.point_labels.update({
                                lab_name: lab_val[inview_idx]
                            })

                if world_coordinates:
                    # Subtract bounding box mins:
                    point_proj = point_proj - self.bounding['min'][:, None]

                # Re-order data by z value
                srt_idx = np.argsort(point_proj[2])
                point_x = point_proj[0][srt_idx]
                point_y = point_proj[1][srt_idx]
                point_z = point_proj[2][srt_idx]

                if point_set_idx == len(points_to_show) - 1:
                    for lab_name, lab_val in self.point_labels.items():
                        self.point_labels.update({
                            lab_name: lab_val[srt_idx]
                        })

                    marker_line_cols = ['rgb(228,26,28)', 'rgb(55,126,184)']
                    for grain_idx in [1, 0]:

                        point_idx = np.where(
                            self.point_labels['grain_idx'] == grain_idx)[0]
                        point_x_i = point_x[point_idx]
                        point_y_i = point_y[point_idx]
                        point_z_i = point_z[point_idx]

                        plot_data.append({
                            'type': 'scatter',
                            'x': point_x_i,
                            'y': point_y_i,
                            'cliponaxis': False,
                            'marker': {
                                'color': point_z_i,
                                'cmin': np.min(point_z),
                                'cmax': np.max(point_z),
                                'colorscale': 'Greys',
                                # 'showscale': True,
                                'size': 9,
                                'line': {
                                    'color': marker_line_cols[grain_idx],
                                    'width': 0.7,
                                }
                            },
                            'xaxis': 'x',
                            'yaxis': 'y',
                            'mode': 'markers',
                        })
                else:

                    plot_data.append({
                        'type': 'scatter',
                        'x': point_x,
                        'y': point_y,
                        'cliponaxis': False,
                        'marker': {
                            'color': 'rgba(255,255,255,0)',
                            'size': 9,
                            'line': {
                                'color': 'gray',
                                'width': 0.7,
                            }
                        },
                        'xaxis': 'x',
                        'yaxis': 'y',
                        'mode': 'markers',
                    })

        if plot_height is None and plot_width is None and pixels_per_unit is None:
            plot_height = 500

        if pixels_per_unit is not None:

            plot_height = pixels_per_unit * self.view_frustum.height
            plot_width = pixels_per_unit * self.view_frustum.width

        else:
            if plot_width is None:
                plot_width = plot_height * self.view_frustum.width / self.view_frustum.height
            elif plot_height is None:
                plot_height = plot_width * self.view_frustum.height / self.view_frustum.width

        print('plot_height: {}'.format(plot_height))
        print('plot_width: {}'.format(plot_width))

        ax_range = [-1.01, 1.01]
        yax_range = ax_range
        xax_range = ax_range
        scale_ratio = self.view_frustum.height / self.view_frustum.width
        print('frustum height: {}'.format(self.view_frustum.height))
        if world_coordinates:
            yax_range = [0, self.view_frustum.height]
            xax_range = [0, self.view_frustum.width]
            scale_ratio = 1

        layout_args_new = {
            'subplot_heights': [plot_height],
            'subplot_widths': [plot_width],
            'x_sep': 10,
            'y_sep': 10,
            'xaxis_label_thickness': 0,
            'yaxis_label_thickness': 0,
            'xaxis_title_thickness': 0,
            'yaxis_title_thickness': 0,
            'extra_margins': {'t': 0, 'r': 0, 'b': 0, 'l': 0},
            'xaxis_props': [{
                'range': xax_range,
                'showline': False,
                'showticklabels': False,
                'ticks': '',
                'zeroline': False,
                'layer': 'below traces',
            }],
            'yaxis_props': [{
                'range': yax_range,
                'anchor': 'x',
                'showline': False,
                'showticklabels': False,
                'ticks': '',
                'zeroline': False,
                'layer': 'below traces',
            }],
            'show_legend': False,
            'show_layout': False,
        }

        # Allow user-specified layout args to overwrite these:
        layout_args = {
            **layout_args_new,
            **layout_args,
        }

        if not world_coordinates:
            layout_args['yaxis_props'][0].update({
                'scaleanchor': 'x',
                'scaleratio': scale_ratio,
            })

        out = {
            'data': plot_data,
            'layout_args': layout_args,
            'target': target,
            'filename': 'geom_proj_{}_{}'.format(self.label, 'wc' if world_coordinates else 'cc'),
            'svg': True,
            'svg_raw': True,
        }

        return out
