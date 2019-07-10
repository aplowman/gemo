'`gemo.geometries.py'

import numpy as np
from vecmaths import geometry


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

    @property
    def corner_coords(self):
        'Get the coordinates of the corners.'

        box_xyz = geometry.get_box_xyz(self.edge_vectors, self.origin)[0]
        return box_xyz
