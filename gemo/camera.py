'`gemo.camera.py` Virtual camera code including a class for an orthographic camera.'

import copy

from gemo.gemo import get_overall_geometry_extrema
from gemo.utils import validate_3d_vector, to_4d_array, point_on_line

import numpy as np
from vecmaths import geometry


def camera_transform(look_at, up, look_from=None):
    """Find the rotation matrix to rotate the model objects into camera space.

    Parameters
    ----------
    look_at : 
        Direction the camera is pointing in.
    up : 
        Orientation of the camera about the `look_at` vector.
    look_from : 
        Position of the camera.

    Returns
    -------
    mat : ndarray of shape (4, 4)

    """

    if look_from is None:
        look_from = [0, 0, 0]

    look_at = validate_3d_vector(look_at)
    up = validate_3d_vector(up)
    look_from = validate_3d_vector(look_from)

    # construct an orthonormal basis:
    u_x = np.cross(look_at, up)
    # (`look_at` and `up` may not be perpendicular)
    u_y = np.cross(u_x, look_at)

    # Normalise:
    u_x = u_x / np.linalg.norm(u_x)
    u_y = u_y / np.linalg.norm(u_y)
    u_z = -look_at / np.linalg.norm(look_at)

    mat = np.vstack([u_x, u_y, u_z])

    trans = np.vstack([
        np.dot(u_x, -look_from),
        np.dot(u_y, -look_from),
        np.dot(u_z, -look_from),
    ])

    mat = np.vstack([
        np.hstack([mat, trans]),
        [0, 0, 0, 1]
    ])

    return mat


class Camera(object):

    def __init__(self, look_at, up, look_from=None):

        self.look_at = validate_3d_vector(look_at).astype(float)
        self.look_from = validate_3d_vector(look_from).astype(float)
        self.up = validate_3d_vector(up).astype(float)

    @property
    def camera_transform(self):
        return camera_transform(self.look_at, self.up, self.look_from)

    @property
    def clipping_planes(self):
        'clipping planes in homogeneous clip space coords Ax + By + Cz + Dw = 0'
        planes = np.array([
            [-1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, -1, 1],
            [0, 0, 1, 1],
        ])
        return planes

    def test_points_in_view(self, points):
        """Test which points are in view. `points` must be in homogeneous
        clipping space coordinates."""
        points_in_planes = np.einsum(
            'ijk,jl->il', self.clipping_planes[:, :, None], points)
        points_in_planes_rnd = np.round(points_in_planes, decimals=7)

        #print('points_in_planes: \n{}\n'.format(points_in_planes))
        points_in = np.all(points_in_planes_rnd >= 0, axis=0)
        return points_in

    def clip_line_segments(self, lines):
        'Clip line segments to the view frustum.'

        distances = np.einsum('ijk,lj->ilk', lines, self.clipping_planes)
        # Which line segments are parallel to which planes
        parallel_idx = distances[:, :, 0] == distances[:, :, 1]

        # print('distances: \n{}\n'.format(distances))
        # print('parallel_idx: \n{}\n'.format(parallel_idx))

        # Find intersection point of (non-parallel) line segments with planes:
        non_para_idx = np.logical_not(parallel_idx)
        intersect_param = np.ones_like(parallel_idx) * np.nan
        intersect_param[non_para_idx] = (-distances[non_para_idx][:, 0]) / (
            distances[non_para_idx][:, 1] - distances[non_para_idx][:, 0])

        with np.errstate(invalid='ignore'):
            intersect_param[intersect_param > 1] = np.nan
            intersect_param[intersect_param < 0] = np.nan

        # intersect_param is now the intersection points (as line segment parameter, t) of each line segment
        # with each plane

        n_lines = lines.shape[0]
        line_seg_in_idx = self.test_points_in_view(
            np.hstack(lines)).reshape((n_lines, 2))

        # print('line_seg_in_idx: \n{}\n'.format(line_seg_in_idx))

        clipped_lines = []
        line_indices = []
        for line_idx in range(n_lines):

            #print('----------------------\nline_idx: {}'.format(line_idx))
            #print('line: \n{}\n'.format(lines[line_idx]))

            #print('line_seg_in_idx[line_idx]: {}'.format(line_seg_in_idx[line_idx]))

            # If both segment ends are inside, keep line and continue
            if all(line_seg_in_idx[line_idx]):
                new_line = lines[line_idx]
                #print('appending new line [INTERNAL]: {}'.format(new_line))
                clipped_lines.append(new_line)
                line_indices.append(line_idx)
                continue

            # Get unique intersection points:
            int_param = intersect_param[line_idx]
            uniq_int_param_all = np.unique(
                np.round(int_param[~np.isnan(int_param)], decimals=8))
            remove_ends_idx = np.logical_or(np.isclose(uniq_int_param_all, 0),
                                            np.isclose(uniq_int_param_all, 1))
            uniq_int_param = uniq_int_param_all[~remove_ends_idx]

            # print('int_param: {}'.format(int_param))
            # print('uniq_int_param: {}'.format(uniq_int_param))

            if uniq_int_param.size:

                # Convert intersection parameters into 4D clip space points:
                start = lines[line_idx][:, 0]
                end = lines[line_idx][:, 1]
                # print('start: {}'.format(start))
                # print('end: {}'.format(end))

                int_points = np.array([point_on_line(start, end, i)
                                       for i in uniq_int_param]).T
                # print('int_points: {}'.format(int_points))

                # Continue with only those intersections point that are within the frustum
                int_points_in_bool = self.test_points_in_view(int_points)
                int_points_in = int_points[:, int_points_in_bool]
                # print('int_points_in_bool: {}'.format(int_points_in_bool))
                # print('int_points_in: {}'.format(int_points_in))

                if int_points_in.size:

                    if not any(line_seg_in_idx[line_idx]):
                        # Both segment ends are outside
                        # Clip each end to the intersection points (should be exactly two):
                        # print('both ends outside')
                        assert int_points_in.shape[1] == 2
                        new_line = int_points_in

                    else:
                        # One segment end is outside
                        # Clip outside point to intersection point (should be exactly one):
                        # print('one end outside')
                        assert int_points_in.shape[1] == 1
                        clip_seg_idx = np.where(~line_seg_in_idx[line_idx])[0][0]
                        # print('clip_seg_idx: {}'.format(clip_seg_idx))
                        new_line = np.copy(lines[line_idx])
                        new_line[:, clip_seg_idx] = int_points_in[:, 0]

                    # print('appending new line [CLIPPED]: {}'.format(new_line))
                    clipped_lines.append(new_line)
                    line_indices.append(line_idx)

        # print('clipped_lines: \n')
        # for i in clipped_lines:
            # print(i)

        if clipped_lines:
            clipped_lines = np.array(clipped_lines)
        else:
            clipped_lines = np.zeros((0, 4, 2))

        # Remove lines segments of zero length (equal start and end points):
        if clipped_lines.size:
            zero_len_idx = np.all(
                np.isclose(clipped_lines[:, :, 0],
                           clipped_lines[:, :, 1]), axis=1)

            clipped_lines = clipped_lines[~zero_len_idx]

        return clipped_lines, line_indices


class OrthographicCamera(Camera):
    """Clipping planes left, right, ... form a box with one corner at (left, bottom,
    -near) and the opposite corner at (right, top, -far)."""

    def __init__(self, look_at, up, width, height, depth, look_from=None):

        super().__init__(look_at, up, look_from)

        self.width = width
        self.height = height
        self.depth = -depth

    @classmethod
    def from_bounding_box(cls, geometry_group, look_at, up, width=None, height=None,
                          depth=None, camera_translate=None):
        """Create the camera from the bounding box of a geometry group.

        Parameters
        ----------
        geometry_group : GeometryGroup
            The geometry group whose bounding box is used to generate the camera.

        """

        # Find the correct `look_from` and frustum dimensions. Since the geometry group
        # centroid is axis aligned, it will change as the geometry group is rotated,
        # so we need the centroid of the rotation geometry group to be the `look_from`
        # vector.
        cam1 = camera_transform(look_at, up)
        cam1_inv = np.linalg.inv(cam1)
        geometry_group = copy.deepcopy(geometry_group)
        geometry_group.transform(cam1[:3, :3])
        look_from = (cam1_inv @ np.append(geometry_group.centroid, [0])[:, None])[:3, 0]
        geometries = {
            'points': geometry_group.points,
            'boxes': geometry_group.boxes,
            'lines': geometry_group.lines,
        }
        extrema = get_overall_geometry_extrema(geometries)
        dims = extrema[:, 1] - extrema[:, 0]

        if not width:
            width = dims[0]
        if not height:
            height = dims[1]
        if not depth:
            depth = dims[2]

        cam2 = camera_transform(look_at, up, look_from)
        cam2_inv = np.linalg.inv(cam2)

        # Translate `look_from` away from the centroid, towards the near plane of the
        # bounding box:
        depth_translate = (cam2_inv @ np.array([[0, 0, depth / 2, 1]]).T)[:3, 0]
        depth_translate = depth_translate - look_from
        new_look_from = np.copy(look_from) + depth_translate

        if camera_translate is not None:
            camera_translate = validate_3d_vector(camera_translate).astype(float)
            translate = (cam2_inv @ np.append(camera_translate, [0])[:, None])[:3, 0]
            new_look_from += translate

        camera = cls(
            look_at,
            up,
            width=width,
            height=height,
            depth=depth,
            look_from=new_look_from,
        )

        return camera

    def get_frustum_camera(self):
        edge_vectors = np.array([
            [self.width, 0, 0],
            [0, self.height, 0],
            [0, 0, self.depth],
        ])
        origin = np.array([
            self.left,
            self.bottom,
            self.near,
        ])
        return (edge_vectors, origin)

    def get_frustum_camera_xyz(self):

        edge, origin = self.get_frustum_camera()
        xyz = geometry.get_box_xyz(edge.astype(float), origin.astype(float))[0]
        return xyz

    def get_frustum_world(self):

        edge_cam, origin_cam = self.get_frustum_camera()

        # to homogeneous coords:
        edge_cam_h = to_4d_array(edge_cam)
        origin_cam_h = np.hstack([origin_cam, 1])[:, None]
        cam_inv = np.linalg.inv(self.camera_transform)

        edge_world = (cam_inv @ edge_cam_h)[:3, :3]
        origin_world = (cam_inv @ origin_cam_h)[:3, 0]

        return (edge_world, origin_world)

    def get_frustum_world_xyz(self):

        edge, origin = self.get_frustum_world()
        xyz = geometry.get_box_xyz(edge, origin)[0]
        return xyz

    @property
    def left(self):
        return - self.width / 2

    @property
    def right(self):
        return self.width / 2

    @property
    def bottom(self):
        return - self.height / 2

    @property
    def top(self):
        return self.height / 2

    @property
    def near(self):
        return 0

    @property
    def far(self):
        return self.depth

    @property
    def projection_transform(self):

        trans = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, (self.near + self.far) / 2],
            [0, 0, 0, 1],
        ])

        scaling = np.array([
            [(2 / self.width), 0, 0, 0],
            [0, (2 / self.height), 0, 0],
            [0, 0, (2 / self.depth), 0],
            [0, 0, 0, 1],
        ])

        proj = scaling @ trans

        return proj
