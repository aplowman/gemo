'`gemo.gemo.py`'

import copy
from pprint import pprint

import numpy as np
from vecmaths import geometry
from spatial_sites import Sites
from spatial_sites.utils import repr_dict

from gemo.geometries import Box, ProjectedBox
from gemo.backends import make_figure_func
from gemo.utils import nest, validate_3d_vector, get_lines_trace


class GeometryGroup(object):

    def __init__(self, points=None, boxes=None, lines=None):

        self.points = self._validate_points(points)
        self.boxes = self._validate_boxes(boxes)
        self.lines = self._validate_lines(lines)

    def __repr__(self):

        points = repr_dict(self.points, indent=4)
        boxes = repr_dict(self.boxes, indent=4)
        lines = repr_dict(self.lines, indent=4)

        indent = ' ' * 4
        out = (
            '{0}(\n'
            '{1}points={2},\n'
            '{1}boxes={3},\n'
            '{1}lines={4},\n'
            ')'.format(
                self.__class__.__name__,
                indent,
                points,
                boxes,
                lines,
            )
        )
        return out

    def __copy__(self):
        points = copy.deepcopy(self.points)
        boxes = copy.deepcopy(self.boxes)
        lines = copy.deepcopy(self.lines)
        return GeometryGroup(points=points, boxes=boxes, lines=lines)

    def _validate_points(self, points):

        if not points:
            return {}

        msg = ('`points` must be a dict whose keys are strings and whose values are '
               '`Sites` objects.')
        if not isinstance(points, dict):
            raise ValueError(msg)

        for val in points.values():
            if not isinstance(val, Sites):
                raise ValueError(msg)

        return points

    def _validate_boxes(self, boxes):

        if not boxes:
            return {}

        msg = ('`boxes` must be a dict whose keys are strings and whose values are '
               '`Box` objects.')
        if not isinstance(boxes, dict):
            raise ValueError(msg)

        for val in boxes.values():
            if not isinstance(val, Box):
                raise ValueError(msg)

        return boxes

    def _validate_lines(self, lines):

        if not lines:
            return {}

        msg = ('`lines` must be a dict whose keys are strings and whose values are '
               '`ndarray`s of shape (N, 3, 2).')
        if not isinstance(lines, dict):
            raise ValueError(msg)

        for val in lines.values():
            if not isinstance(val, np.ndarray) or val.shape[1:] != (3, 2):
                raise ValueError(msg)

        return lines

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
                uniq = [points_set.labels[i].unique_values for i in group_names]

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
                            style_name: style_vals[i[group_names.index(label_name)]]
                            for label_name, label_styles in all_styles.items()
                            for style_name, style_vals in label_styles.items()
                        }
                    })

        boxes = []
        for box_name, box in self.boxes.items():
            coords = box.edge_vertices
            boxes.append({
                'name': box_name,
                'x': coords[0],
                'y': coords[1],
                'z': coords[2],
            })

        lines = []
        for ln_name, ln in self.lines.items():
            ln_trace = get_lines_trace(ln)
            lines.append({
                'name': ln_name,
                'x': ln_trace[0],
                'y': ln_trace[1],
                'z': ln_trace[2],
            })

        data = {
            'points': points,
            'boxes': boxes,
            'lines': lines,
        }

        return data

    def copy(self):
        return self.__copy__()

    def show(self, group_points=None, layout_args=None, target='interactive',
             backend='plotly'):

        group_points = self._validate_points_grouping(group_points)
        plot_data = self._get_plot_data(group_points)
        fig = make_figure_func[backend](plot_data, layout_args)

        return fig

    def rotate(self, rot_mat):
        'Rotate according to a rotation matrix.'

        # Rotate points coords:
        points = {}
        for points_name, points_set in self.points.items():
            points_set.transform(rot_mat)
            points.update({points_name: points_set})
        self.points = points

        # Rotate boxes:
        boxes = {}
        for box_name, box in self.boxes.items():
            box.rotate(rot_mat)
            boxes.update({box_name: box})
        self.boxes = boxes

        # Rotate lines:
        for lines_name, lines in self.lines.items():
            self.lines[lines_name] = rot_mat @ lines

    @property
    def bounding_coordinates(self):
        'Get the orthogonal bounding box minima and maxima in each dimension.'

        # Concatenate all points and box coordinates:
        points = np.hstack([i._coords for i in self.points.values()])
        box_coords = np.hstack([i.vertices for i in self.boxes.values()])
        line_coords = np.hstack(self.lines.values())
        all_coords = np.hstack([points, box_coords, line_coords])

        out = np.array([
            np.min(all_coords, axis=1),
            np.max(all_coords, axis=1)
        ]).T

        return out
