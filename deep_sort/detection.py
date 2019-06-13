# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    feature : array_like
        A feature vector that describes the object contained in this image.
    label  : array_like
        Label class instance which contains predicted class labels and score.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    label  : array_like
        Label class instance which contains predicted class labels and score.

    """

    def __init__(self, tlwh, feature, label):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.label = label

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
