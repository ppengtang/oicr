# --------------------------------------------------------
# Online Instance Classifier Refinement
# Copyright (c) 2016 HUST MCLAB
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Tang
# --------------------------------------------------------

"""The layer used during training to get proposal labels for classifier refinement.

OICRLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import scipy.sparse
import yaml

from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class OICRLayer(caffe.Layer):
    """get proposal labels used for online instance classifier refinement."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        if len(bottom) != 3:
            raise Exception("The number of bottoms should be 3!")

        if len(top) != 2:
            raise Exception("The number of tops should be 2!")

        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("bottom[0].data.shape[0] must equal to bottom[1].data.shape[0]")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1)

        top[1].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        boxes = bottom[0].data[:, 1:]
        cls_prob = bottom[1].data
        if cls_prob.shape[1] == self._num_classes:
            cls_prob = cls_prob[:, 1:]
        im_labels = bottom[2].data

        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels)
        labels, rois, cls_loss_weights = _sample_rois(boxes, proposals, self._num_classes)

        assert rois.shape[0] == boxes.shape[0]

        # classification labels
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        top[1].reshape(*cls_loss_weights.shape)
        top[1].data[...] = cls_loss_weights        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i]
            max_index = np.argmax(cls_prob_tmp)

            if DEBUG:
                print 'max_index:', max_index, 'cls_prob_tmp:', cls_prob_tmp[max_index]

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1))) 
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores, 
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def _sample_rois(all_rois, proposals, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    if DEBUG:
        print "number of fg:", len(fg_inds), 'number of bg:', len(bg_inds) 

    labels[bg_inds] = 0

    rois = all_rois

    return labels, rois, cls_loss_weights
