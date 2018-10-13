# --------------------------------------------------------
# Proposal Cluster Learning
# Copyright (c) 2018 HUST MCLAB
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Tang
# --------------------------------------------------------

"""The layer used during training to perform proposal cluster learning.

OICRLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import scipy.sparse
import yaml
from sklearn.cluster import KMeans

from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class OICRLayer(caffe.Layer):
    """get proposal labels used for online instance classifier refinement."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        if len(bottom) != 4:
            raise Exception("The number of bottoms should be 4!")

        if len(top) != 7 and len(top) != 8:
            raise Exception("The number of tops should be 7 or 8!")

        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("bottom[0].data.shape[0] must equal to bottom[1].data.shape[0]")

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        top[0].reshape(1)

        top[1].reshape(1)

        top[2].reshape(1)

        top[3].reshape(1)

        top[4].reshape(1)

        top[5].reshape(1)

        top[6].reshape(1)

        if len(top) == 8:
            top[7].reshape(1, self._num_classes)
            

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        boxes = bottom[0].data[:, 1:]
        cls_prob = bottom[1].data
        if cls_prob.shape[1] == self._num_classes:
            cls_prob = cls_prob[:, 1:]
        im_labels = bottom[2].data
        cls_prob_new = bottom[3].data

        proposals = _get_graph_centers(boxes.copy(), cls_prob.copy(), 
            im_labels.copy())

        labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
            pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes.copy(), 
                proposals, im_labels.copy(), cls_prob_new.copy())

        # proposal labels
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # loss weights for each proposal
        top[1].reshape(*cls_loss_weights.shape)
        top[1].data[...] = cls_loss_weights        

        # proposal cluster ID
        top[2].reshape(*gt_assignment.shape)
        top[2].data[...] = gt_assignment

        # proposal cluster labels
        top[3].reshape(*pc_labels.shape)
        top[3].data[...] = pc_labels

        # average pooled probability for each proposal cluster
        top[4].reshape(*pc_probs.shape)
        top[4].data[...] = pc_probs

        # number of proposals in each proposal cluster
        top[5].reshape(*pc_count.shape)
        top[5].data[...] = pc_count

        # loss weights for each proposal cluster
        top[6].reshape(*img_cls_loss_weights.shape)
        top[6].data[...] = img_cls_loss_weights

        if len(top) == 8:
            # add the background class to the image level label vector
            im_labels_real = np.hstack((np.array([[1]], 
                dtype=im_labels.dtype), im_labels))
            top[7].reshape(*im_labels_real.shape)
            top[7].data[...] = im_labels_real

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    kmeans = KMeans(n_clusters=cfg.TRAIN.NUM_KMEANS_CLUSTER, 
        random_state=cfg.RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index 

def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    overlaps = bbox_overlaps(
        np.ascontiguousarray(boxes, dtype=np.float),
        np.ascontiguousarray(boxes, dtype=np.float))

    return (overlaps > iou_threshold).astype(np.float32)

def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg.TRAIN.GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)\
                [-1:(-1 - min(len(gt_scores_tmp), cfg.TRAIN.MAX_PC_NUM)):-1]
            
            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores, 
                gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack((gt_classes, 
                (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32)))

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
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

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    # ig_inds = np.where(max_overlaps < 0.1)[0]
    # cls_loss_weights[ig_inds] = 0.0

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])

    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights
