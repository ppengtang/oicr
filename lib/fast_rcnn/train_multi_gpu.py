# --------------------------------------------------------
# Written by Bharat Singh
# Modified version of py-R-FCN
#
# Modified by Peng Tang for PCL
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from multiprocessing import Process

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, gpu_id,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.gpu_id = gpu_id

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb, gpu_id)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        if self.gpu_id == 0:
            net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def getSolver(self):
        return self.solver

def solve(proto, roidb, pretrained_model, gpus, uid, rank, output_dir, max_iter):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)
    cfg.GPU_ID = gpus[rank]

    solverW = SolverWrapper(proto, roidb, output_dir,rank,pretrained_model)
    solver = solverW.getSolver()
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    count = 0
    while count < max_iter:
        solver.step(cfg.TRAIN.SNAPSHOT_ITERS)
        if rank == 0:
            solverW.snapshot()
        count = count + cfg.TRAIN.SNAPSHOT_ITERS

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   At least one foreground label
        labels = entry['labels']
        boxes = entry['boxes']
        valid = np.sum(labels) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb


def train_net_multi_gpu(solver_prototxt, roidb, output_dir, pretrained_model, max_iter, gpus):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_prototxt, roidb, pretrained_model, gpus, uid, rank, output_dir, max_iter))
        p.daemon = False
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
