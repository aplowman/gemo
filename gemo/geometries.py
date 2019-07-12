'`gemo.geometries.py'

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
        self.origin = origin

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

    def _validate_edge_labels(self, edge_labels):

        if edge_labels is not None:
            if not isinstance(edge_labels, list) or len(edge_labels) != 3:
                msg = '`edge_labels` must be a list of length three.'
                raise ValueError(msg)

        return edge_labels

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):

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

        self._origin = origin

    def rotate(self, rot_mat):
        'Rotate the box.'

        self.edge_vectors = rot_mat @ self.edge_vectors
        self.origin = rot_mat @ self.origin

    @property
    def vertices(self):
        'Get the coordinates of the corners.'
        return geometry.get_box_corners(self.edge_vectors, self.origin)[0]

    @property
    def edge_idx_traces(self):
        'Return groups of vertex indices that form continuous traces.'
        return [
            [
                [0, 1],
                [1, 4],
                [4, 2],
                [2, 0],
            ],
            [
                [3, 5],
                [5, 7],
                [7, 6],
                [6, 3],
            ],
            [
                [0, 3],
            ],
            [
                [1, 5],
            ],
            [
                [4, 7],
            ],
            [
                [2, 6],
            ],
        ]

    @property
    def edge_idx(self):
        'Get an array of indices of vertices that form edges.'
        return np.concatenate(self.edge_idx_traces)

    def get_edge_vectors(self):
        'Get vectors that translate from initial to final vertex of each edge.'
        pass

    def get_face_planes(self):
        'Get the planes that define the faces of the box.'

        verts = self.vertices
        faces_verts = [
            verts[:, [0, 1, 4, 2, 0]],
            verts[:, [3, 5, 7, 6, 3]],
            verts[:, [0, 1, 5, 3, 0]],
            verts[:, [2, 4, 7, 6, 2]],
            verts[:, [0, 2, 6, 3, 0]],
            verts[:, [1, 4, 7, 5, 1]],
        ]

        planes = []
        for face in faces_verts:
            point = face[:, 0]
            norm = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
            planes.append(Plane(point=point, normal_vector=norm))

        return planes

    @property
    def edge_vertices(self):
        'Get an array of vertices that form a trace for plotting.'

        all_verts = []
        for i in self.edge_idx_traces:
            verts_idx = np.concatenate(i)
            verts_idx = np.concatenate([verts_idx[:2],
                                        np.roll(verts_idx, -1)[2::2]])
            verts_idx = np.append(verts_idx, verts_idx[0])
            all_verts.append(self.vertices[:, verts_idx])

        # Insert np.nan columns to represent a break in the trace:
        all_verts = intersperse(all_verts, np.array([[np.nan] * 3]).T)
        vertices = np.hstack(all_verts)

        return vertices


class ProjectedBox(object):
    'A parallelepiped that has been projected into 2D.'

    def __init__(self, box, view_frustum):
        """
        Parameters
        ----------
        box : Box
        view_frustum : ViewFrustum

        """

        self.box = box
        self.view_frustum = view_frustum

        self._set_projection_attributes()

    def _set_projection_attributes(self):

        verts_homo = np.vstack([self.box.vertices, np.ones(8)])
        verts_proj = self.view_frustum.projection_matrix @ verts_homo

        # Which vertices are within the frustum:
        in_view = np.all(np.logical_and(
            verts_proj <= 1, verts_proj >= -1), axis=0)

        self.vertices_proj = verts_proj

        frustum_planes = self.view_frustum.box.get_face_planes()

        for edge_group in self.box.edge_idx_traces:

            for edge_idx in edge_group:
                print('edge_idx: {}'.format(edge_idx))

                plane_intersects = []
                for plane in frustum_planes:
                    #print('plane: {}'.format(plane))
                    intersect = plane.line_segment_intersection(
                        self.box.vertices[:, edge_idx[0]],
                        self.box.vertices[:, edge_idx[1]],
                    )
                    print('intersect: {}'.format(intersect))
        # planes = ...
        # For each edge (line segment), find where (or whether) it intersects
        # with the viewing frustum...
        #   - for each plane:
        #       - if it doesn't intersect:
        #             if both vertices are outside the viewing frustum:
        #                 discard the edge
        #             else:
        #                 keep the whole edge.
        #       - if it intersects once:
        #             trim the start or end of the edge
        #       - else:
        #             trim both ends of the edge

        # set edge_vertices and edge_vertices_proj
