# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import numpy as np

from misc import voc_eval


def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
        if cls == '__background__':
            continue
        print('Writing {} VOC results file'.format(cls))
        filename = self._get_voc_results_file_template().format(cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
        self._devkit_path,
        'VOC' + self._year,
        'Annotations',
        '{:s}.xml')
    imagesetfile = os.path.join(
        self._devkit_path,
        'VOC' + self._year,
        'ImageSets',
        'Main',
        self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
        if cls == '__background__':
            continue
        filename = self._get_voc_results_file_template().format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


def evaluate_detections(self, all_boxes, output_dir=None):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['cleanup']:
        for cls in self._classes:
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            os.remove(filename)