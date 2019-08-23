'`gemo.gemo.py`'

import copy
from pprint import pprint

import numpy as np
from vecmaths import geometry
from spatial_sites import Sites
from spatial_sites.utils import repr_dict

from gemo.geometries import Box
from gemo.backends import make_figure_func
from gemo.utils import nest, validate_3d_vector, get_lines_trace, get_box_edges


def get_plot_data(points, boxes, lines, group_points, style_points, dimension):
    'Collate data for plotting.'

    if not style_points:
        style_points = {}

    points_dat = []
    for points_name, points_set in points.items():

        pts_style = style_points.get(points_name, {})

        if points_name not in group_points:
            pts_dat = {
                'x': points_set.get_components(0),
                'y': points_set.get_components(1),
                'name': points_name,
                'styles': pts_style,
            }
            if dimension == 3:
                pts_dat.update({
                    'z': points_set.get_components(2),
                })
            points_dat.append(pts_dat)

        elif points_name in group_points:
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
                styles = copy.deepcopy(pts_style)
                styles.update({
                    style_name: style_vals[i[group_names.index(label_name)]]
                    for label_name, label_styles in all_styles.items()
                    for style_name, style_vals in label_styles.items()
                })
                pts_dat = {
                    'name': group_name_fmt.format(*i),
                    'x': pts[0],
                    'y': pts[1],
                    'styles': styles,
                }
                if dimension == 3:
                    pts_dat.update({'z': pts[2]})
                points_dat.append(pts_dat)

    boxes_dat = []
    for box_name, box in boxes.items():
        edges_trace = get_lines_trace(box.edges)
        box_dat = {
            'name': box_name,
            'x': edges_trace[0],
            'y': edges_trace[1],
        }
        if dimension == 3:
            box_dat.update({'z': edges_trace[2]})
        boxes_dat.append(box_dat)

    lines_dat = []
    for ln_name, ln in lines.items():
        ln_trace = get_lines_trace(ln)
        ln_dat = {
            'name': ln_name,
            'x': ln_trace[0],
            'y': ln_trace[1],
        }
        if dimension == 3:
            ln_dat.update({'z': ln_trace[2]})
        lines_dat.append(ln_dat)

    data = {
        'points': points_dat,
        'boxes': boxes_dat,
        'lines': lines_dat,
    }

    return data


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

    def validate_points_grouping(self, group_dict):

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

    def _get_plot_data(self, group_points, style_points):
        return get_plot_data(self.points, self.boxes, self.lines, group_points,
                             style_points, dimension=3)

    def copy(self):
        return self.__copy__()

    def show(self, group_points=None, style_points=None, layout_args=None,
             target='interactive', backend='plotly'):

        print('GG.show: layout_args: {}'.format(layout_args))

        group_points = self.validate_points_grouping(group_points)
        plot_data = self._get_plot_data(group_points, style_points)
        fig = make_figure_func[backend](plot_data, layout_args, dimension=3)

        return fig

    def project(self, camera):
        return GeometryGroupProjection(self, camera)

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

    @property
    def box_edges(self):
        'Get all box edges as couples of vertices.'
        return {name: get_box_edges(box.vertices) for name, box in self.boxes.items()}


class GeometryGroupProjection(object):

    box_styles = {
        'mode': 'lines',
    }

    proj_line_styles = {
        'mode': 'lines',
    }

    def __init__(self, geometry_group, camera):
        self.geometry_group = geometry_group
        self.camera = camera
        self.clipped_geometries = self._clip_geometries()

    def _clip_geometries(self):
        'Clip all geometries to within the view frustum.'

        verts_homo = self._get_homogeneous_verts()
        verts_cam = self._to_camera_space(verts_homo)
        verts_clip = self._to_clip_space(verts_cam)
        clipped_verts_clip = self._clip_vertices(verts_clip)

        return clipped_verts_clip

    def _get_homogeneous_verts(self):
        'Copy geometries and add homogeneous coordinates.'

        points_h = {}
        for name, pts in self.geometry_group.points.items():
            pts_new = pts.copy()
            points_h.update({name: pts_new.to_homogeneous()})

        lines_h = {name: np.concatenate([i, np.ones((i.shape[0], 1, 2))], axis=1)
                   for name, i in self.geometry_group.lines.items()}

        boxes_h = {}
        for name, box in self.geometry_group.boxes.items():
            box_new = box.copy()
            box_new.edges = np.concatenate(
                [box.edges, np.ones((box.edges.shape[0], 1, 2))], axis=1)
            boxes_h.update({name: box_new})

        # print('boxes_h: {}'.format(boxes_h))

        out = {
            'points': points_h,
            'lines': lines_h,
            'boxes': boxes_h,
        }

        return out

    def _to_camera_space(self, verts_world):
        'Transform vertices from world space to camera space.'

        cam_trans = self.camera.camera_transform
        points_h_cam = {name: cam_trans @ i for name, i in verts_world['points'].items()}
        lines_h_cam = {name: cam_trans @ i for name, i in verts_world['lines'].items()}

        boxes_h_cam = {}
        for name, box in verts_world['boxes'].items():
            box.edges = cam_trans @ box.edges
            boxes_h_cam.update({name: box})

        out = {
            'points': points_h_cam,
            'lines': lines_h_cam,
            'boxes': boxes_h_cam,
        }

        return out

    def _to_clip_space(self, verts_cam):
        'Transform vertices from camera space to clip space.'

        proj_trans = self.camera.projection_transform
        points_h_clip = {name: proj_trans @ i for name, i in verts_cam['points'].items()}
        lines_h_clip = {name: proj_trans @ i for name, i in verts_cam['lines'].items()}

        boxes_h_clip = {}
        for name, box in verts_cam['boxes'].items():
            box.edges = proj_trans @ box.edges
            boxes_h_clip.update({name: box})

        out = {
            'points': points_h_clip,
            'lines': lines_h_clip,
            'boxes': boxes_h_clip,
        }

        return out

    def _clip_vertices(self, verts_clip):
        'Clip geometry vertices to within the view frustum.'

        cpoints_h_clip = {name: i[self.camera.test_points_in_view(i._coords)]
                          for name, i in verts_clip['points'].items()}

        clines_h_clip = {name: self.camera.clip_line_segments(i)[0]
                         for name, i in verts_clip['lines'].items()}

        cboxes_h_clip = {}
        for name, box in verts_clip['boxes'].items():
            box.edges = self.camera.clip_line_segments(box.edges)[0]
            cboxes_h_clip.update({name: box})

        out = {
            'points': cpoints_h_clip,
            'lines': clines_h_clip,
            'boxes': cboxes_h_clip,
        }
        return out

    def _get_lines_traces(self, lines_dict):
        'Get traces suitable for plotting.'
        traces = {}
        for k, v in lines_dict.items():
            traces.update({
                k: [get_lines_trace(i) for i in v]
            })
        return traces

    def _get_projected_verts(self):
        'Prepare the 2D projection vertices.'

        # "Perspective divide":
        cpoints_h_ndc = {name: i / i.w
                         for name, i in self.clipped_geometries['points'].items()}
        clines_h_ndc = {name: i / i[:, 3][:, None]
                        for name, i in self.clipped_geometries['lines'].items()}

        cboxes_h_ndc = {}
        for name, box in self.clipped_geometries['boxes'].items():
            box_new = box.copy()
            box_new.edges = box.edges / box.edges[:, 3][:, None]
            cboxes_h_ndc.update({name: box_new})

        # Translate and scale back to world coordinates # TODO change this to a matrix:
        # TODO: perhaps this can be 3x3 matrix (not if it translates...)?
        trans = np.array([[1, 1, 0, 0]]).T
        scale = np.array([[self.camera.width/2, self.camera.height/2, 1, 1]]).T

        cpoints_h_world = {name: (i + trans) * scale for name, i in cpoints_h_ndc.items()}
        clines_h_world = {name: (i + trans) * scale for name, i in clines_h_ndc.items()}

        cboxes_h_world = {}
        for name, box in cboxes_h_ndc.items():
            box.edges = (box.edges + trans) * scale
            cboxes_h_world.update({name: box})

        out = {
            'points': cpoints_h_world,
            'lines': clines_h_world,
            'boxes': cboxes_h_world,
        }

        return out

    def preview(self, group_points=None, layout_args=None, backend='plotly'):
        'Show 3D figure with clipped geometries and view frustum.'

        # Call `show` method on the geometry group with some additional geometries to
        # represent the camera.

        print('GGP.preview: layout_args: {}'.format(layout_args))

        geom_group = self.geometry_group.copy()

        frustum_edge, frustum_origin = self.camera.get_frustum_world()  # TODO: return Box
        #print('making frustum box')
        geom_group.boxes.update({
            'Frustum': Box(edge_vectors=frustum_edge, origin=frustum_origin)
        })

        geom_group.points.update({
            'Camera origin': Sites(coords=self.camera.look_from.reshape(3, 1)),
        })

        cam_look_from_line = np.tile(self.camera.look_from.reshape((3, 1)), (1, 2))

        cam_look_at_vec = np.copy(cam_look_from_line)
        cam_look_at_vec[:, 1] += self.camera.look_at

        cam_look_up_vec = np.copy(cam_look_from_line)
        cam_look_up_vec[:, 1] += self.camera.up

        geom_group.lines.update({
            'Camera (look at)': cam_look_at_vec.reshape((1, 3, 2)),
            'Camera (up)': cam_look_up_vec.reshape((1, 3, 2)),
        })

        out = geom_group.show(
            group_points=group_points,
            layout_args=layout_args,
            backend=backend,
        )
        return out

    def show(self, group_points=None, style_points=None, layout_args=None, backend='plotly'):
        'Show 2D projection.'

        group_points = self.geometry_group.validate_points_grouping(group_points)
        projected_verts = self._get_projected_verts()

        # print('projected_verts')
        # pprint(projected_verts)

        plot_data = get_plot_data(projected_verts['points'], projected_verts['boxes'],
                                  projected_verts['lines'], group_points, style_points,
                                  dimension=2)

        # print('plot_data')
        # pprint(plot_data)

        # For axes labels:
        u_x = self.camera.camera_transform[0, :3]
        u_y = self.camera.camera_transform[1, :3]

        layout_args = layout_args or {}
        layout_args.update({
            'xaxis': {
                'title': '({:.1f}, {:.1f}, {:.1f}) ➔'.format(*u_x),
            },
            'yaxis': {
                'title': '({:.1f}, {:.1f}, {:.1f}) ➔'.format(*u_y),
            },
        })

        fig = make_figure_func[backend](plot_data, layout_args, dimension=2)

        return fig
