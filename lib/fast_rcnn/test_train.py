# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for Online Instance Classifier Refinement
# --------------------------------------------------------

"""Test an OICR network on an imdb (image database), for trainval set (CorLoc)."""

from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        im_shapes: the list of image shapes
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois_blob_real = []

    for i in xrange(len(im_scale_factors)):
        rois, levels = _project_im_rois(im_rois, np.array([im_scale_factors[i]]))
        rois_blob = np.hstack((levels, rois))
        rois_blob_real.append(rois_blob.astype(np.float32, copy=False))

    return rois_blob_real

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1
        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    
    return blobs, im_scale_factors

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    for i in xrange(len(blobs['data'])):
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'][i] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            blobs['rois'][i] = blobs['rois'][i][index, :]
            boxes_tmp = boxes[index, :]
        else:
            boxes_tmp = boxes

        # reshape network inputs
        net.blobs['data'].reshape(*(blobs['data'][i].shape))
        net.blobs['rois'].reshape(*(blobs['rois'][i].shape))
        
        blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False),
                                rois=blobs['rois'][i].astype(np.float32, copy=False))

        scores_tmp = blobs_out['ic_prob'] + blobs_out['ic_prob1'] + blobs_out['ic_prob2']
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes_tmp, (1, scores_tmp.shape[1]))

        if cfg.TEST.USE_FLIPPED:
            blobs['data'][i] = blobs['data'][i][:, :, :, ::-1]
            width = blobs['data'][i].shape[3]
            oldx1 = blobs['rois'][i][:, 1].copy()
            oldx2 = blobs['rois'][i][:, 3].copy()
            blobs['rois'][i][:, 1] = width - oldx2 - 1
            blobs['rois'][i][:, 3] = width - oldx1 - 1
            assert (blobs['rois'][i][:, 3] >= blobs['rois'][i][:, 1]).all()

            net.blobs['data'].reshape(*(blobs['data'][i].shape))
            net.blobs['rois'].reshape(*(blobs['rois'][i].shape))

            blobs_out = net.forward(data=blobs['data'][i].astype(np.float32, copy=False),
                                    rois=blobs['rois'][i].astype(np.float32, copy=False))

            scores_tmp += blobs_out['ic_prob'] + blobs_out['ic_prob1'] + blobs_out['ic_prob2']

        if cfg.DEDUP_BOXES > 0:
            # Map scores and predictions back to the original set of boxes
            scores_tmp = scores_tmp[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

        if i == 0:        
            scores = np.copy(scores_tmp)
        else:
            scores += scores_tmp

    scores /= len(blobs['data']) * (1. + cfg.TEST.USE_FLIPPED)

    return scores[:, 1:], pred_boxes[:, 4:]

def test_net_train(net, imdb):
    """Test an OICR network on an image database, 
    and generate pseudo ground truths for training fast rcnn."""
    num_images = len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # thresh = 0.1 * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    images_real = np.zeros((num_images,), dtype=object)
    gt = np.zeros((num_images, ), dtype=object)
    roidb = imdb.roidb
    
    scores_all = []
    boxes_all = []
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, roidb[i]['boxes'])
        _t['im_detect'].toc()
        scores_all.append(scores)
        boxes_all.append(boxes)

        _t['misc'].tic()
        for j in xrange(imdb.num_classes):
            index = np.argmax(scores[:, j])
            all_boxes[j][i] = \
                np.hstack((boxes[index, j*4:(j+1)*4].reshape(1, -1), 
                           np.array([[scores[index, j]]])))

        gt_tmp = {'aeroplane' : np.empty((0, 4), dtype=np.float32), 
                  'bicycle' : np.empty((0, 4), dtype=np.float32), 
                  'bird' : np.empty((0, 4), dtype=np.float32), 
                  'boat' : np.empty((0, 4), dtype=np.float32), 
                  'bottle' : np.empty((0, 4), dtype=np.float32), 
                  'bus' : np.empty((0, 4), dtype=np.float32), 
                  'car' : np.empty((0, 4), dtype=np.float32), 
                  'cat' : np.empty((0, 4), dtype=np.float32), 
                  'chair' : np.empty((0, 4), dtype=np.float32), 
                  'cow' : np.empty((0, 4), dtype=np.float32), 
                  'diningtable' : np.empty((0, 4), dtype=np.float32), 
                  'dog' : np.empty((0, 4), dtype=np.float32), 
                  'horse' : np.empty((0, 4), dtype=np.float32), 
                  'motorbike' : np.empty((0, 4), dtype=np.float32), 
                  'person' : np.empty((0, 4), dtype=np.float32), 
                  'pottedplant' : np.empty((0, 4), dtype=np.float32), 
                  'sheep' : np.empty((0, 4), dtype=np.float32), 
                  'sofa' : np.empty((0, 4), dtype=np.float32), 
                  'train' : np.empty((0, 4), dtype=np.float32), 
                  'tvmonitor':np.empty((0, 4), dtype=np.float32)}
        tmp_idx = np.where(roidb[i]['labels'][0][:imdb.num_classes])[0]

        for j in xrange(len(tmp_idx)):
            idx_real = np.argmax(scores[:, tmp_idx[j]])
            gt_tmp[imdb.classes[tmp_idx[j]]] = np.array([boxes[idx_real, tmp_idx[j]*4+1], 
                                                         boxes[idx_real, tmp_idx[j]*4], 
                                                         boxes[idx_real, tmp_idx[j]*4+3],
                                                         boxes[idx_real, tmp_idx[j]*4+2]], dtype=np.float32)
            gt_tmp[imdb.classes[tmp_idx[j]]] += 1

        gt[i] = {'gt' : gt_tmp}

        images_real[i] = imdb.image_index[i]
        
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
        
    model_save_gt = {'images' : images_real, 'gt' : gt}
    sio.savemat('{}_gt.mat'.format(imdb.name), model_save_gt)

    dis_file = os.path.join(output_dir, 'discovery.pkl')
    with open(dis_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
        
    dis_file_all = os.path.join(output_dir, 'discovery_all.pkl')
    results_all = {'scores_all' : scores_all, 'boxes_all' : boxes_all}
    with open(dis_file_all, 'wb') as f:
        cPickle.dump(results_all, f, cPickle.HIGHEST_PROTOCOL)

    imdb.evaluate_discovery(all_boxes, output_dir)
