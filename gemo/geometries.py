'`gemo.geometries.py'

import copy

import numpy as np
from vecmaths import geometry

from gemo.utils import intersperse, validate_3d_vector


class Plane(object):

    def __init__(self, point, normal_vector):

        self.point = self._validate_point(point)
        self.normal_vector = self._validate_normal(normal_vector)

    def _validate_normal(self, normal):
        normal = validate_3d_vector(normal)
        normal = normal / np.linalg.norm(normal)
        return normal

    def _validate_point(self, point):
        return validate_3d_vector(point)

    def line_segment_intersection(self, line_start, line_end):
        """Find the intersection point with a line segment. Inspired by
        https://math.stackexchange.com/q/47594/672781.

        Parameters
        ----------
        line_start : list or ndarray of length 3
            Starting point of the line segment.
        line_end : list or ndarray of length 3
            Terminating point of the line segment.

        Returns
        -------
        intersection : bool or ndarray of size 3
            If True, the line segment lies in the plane. If False, the line
            segment does not intersect the plane. Otherwise, the point of
            intersection is returned as an array.

        """

        ls = validate_3d_vector(line_start)
        le = validate_3d_vector(line_end)

        # Get perpendicular distances of line points from plane:
        ds = np.dot(self.normal_vector, ls - self.point)
        de = np.dot(self.normal_vector, le - self.point)

        if np.isclose(abs(ds) + abs(de), 0):
            # Line segment lies in plane
            intersection = True

        elif ds * de > 0:
            # Line segment does not intersect plane
            intersection = False

        elif ds * de <= 0:
            # Line segment intersects plane (or starts/ends in plane)

            # Get unit vector along line segment:
            line = le - ls
            lu = line / np.linalg.norm(line)
            cos_theta = np.dot(self.normal_vector, lu)

            # If cos_theta == 0, the line segment is parallel to the plane,
            # in which case, it either lies in the plane (returned above),
            # or it does not intersect the plane (returned above).

            intersection = le - (lu * (de / cos_theta))

        return intersection


class Box(object):

    def __init__(self, edge_vectors, origin=None, edge_labels=None):

        self.edge_vectors = self._validate_edge_vectors(edge_vectors)
        self.edge_labels = self._validate_edge_labels(edge_labels)
        self._origin = self._validate_origin(origin)

        # print('box init')
        # print('edge_vectors: \n{}\n'.format(edge_vectors))
        # print('origin: \n{}\n'.format(origin))
        # print('edge_labels: \n{}\n'.format(edge_labels))

        self.edges = self._get_edges()

    @classmethod
    def from_extrema(cls, mins, maxes):
        'Generate from minimum and maximum coordinates in each dimension.'
        edge_vectors = np.diag(maxes - mins)
        return cls(edge_vectors, origin=mins)

    def __repr__(self):

        indent = ' ' * 4

        edge_vecs = '{!r}'.format(self.edge_vectors).replace(
            '\n', '\n' + indent + ' ' * len('edge_vectors='))

        origin = '{!r}'.format(self.origin).replace(
            '\n', '\n' + indent + ' ' * len('origin='))

        out = (
            '{0}(\n'
            '{1}edge_vectors={2},\n'
            '{1}edge_labels={3!r},\n'
            '{1}origin={4},\n'
            ')'.format(
                self.__class__.__name__,
                indent,
                edge_vecs,
                self.edge_labels,
                origin
            )
        )
        return out

    def _validate_edge_vectors(self, edge_vectors):

        if not isinstance(edge_vectors, np.ndarray):
            edge_vectors = np.array(edge_vectors)

        if edge_vectors.shape != (3, 3):
            msg = '`edge_vectors` must have shape (3, 3), not {}'
            raise ValueError(msg.format(edge_vectors.shape))

        return edge_vectors

    def _validate_origin(self, origin):

        if origin is None:
            origin = np.zeros((3, 1))
        else:
            if isinstance(origin, list):
                origin = np.array(origin)
            elif not isinstance(origin, np.ndarray):
                raise ValueError('`origin` must be a list or ndarray.')
            origin = np.squeeze(origin)[:, None]

        if origin.size != 3:
            msg = '`origin` must have size 3, not {}'
            raise ValueError(msg.format(origin.size))

        return origin

    def _validate_edge_labels(self, edge_labels):

        #print('_validate_edge_labels:: edge_labels: {}'.format(edge_labels))

        if edge_labels is not None:
            if not isinstance(edge_labels, list) or len(edge_labels) != 3:
                msg = '`edge_labels` must be a list of length three.'
                raise ValueError(msg)

        return edge_labels

    def _get_edges(self):
        'Get vertices fo line segments representing the box edges.'

        verts = self.vertices
        edges = [
            verts[:, [0, 1]],
            verts[:, [1, 4]],
            verts[:, [4, 2]],
            verts[:, [2, 0]],
            verts[:, [3, 5]],
            verts[:, [5, 7]],
            verts[:, [7, 6]],
            verts[:, [6, 3]],
            verts[:, [1, 5]],
            verts[:, [3, 0]],
            verts[:, [4, 7]],
            verts[:, [6, 2]],
        ]
        return np.array(edges)

    @property
    def origin(self):
        return self._origin

    @property
    def vertices(self):
        'Get the coordinates of the corners.'
        # print('vertices::')
        #print('self.edge_vectors: \n{}\n'.format(self.edge_vectors))
        #print('self.origin: \n{}\n'.format(self.origin))
        return geometry.get_box_corners(self.edge_vectors, self.origin)[0]

    @property
    def centroid(self):
        return np.mean(self.vertices, axis=1)
