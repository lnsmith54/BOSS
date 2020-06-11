#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to create SSL splits from a dataset.
"""

import json
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange, tqdm

from libml import data as libml_data
from libml import utils

flags.DEFINE_integer('seed', 0, 'Random seed to use, 0 for no shuffling.')
flags.DEFINE_integer('size', 0, 'Size of labelled set.')

FLAGS = flags.FLAGS


def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']


def main(argv):
    assert FLAGS.size
    argv.pop(0)
    if any(not tf.gfile.Exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])
    target = '%s.%d@%d' % (argv[0], FLAGS.seed, FLAGS.size)
    if tf.gfile.Exists(target):
        raise FileExistsError('For safety overwriting is not allowed', target)
    input_files = argv[1:]
    print("=> input_files= ", input_files)
    count = 0
    id_class = []
    class_id = defaultdict(list)
    print('Computing class distribution')
    dataset = tf.data.TFRecordDataset(input_files).map(get_class, 4).batch(1 << 10)
    it = dataset.make_one_shot_iterator().get_next()
####  Prototype code
    if (FLAGS.seed ==0):
        label=  [35, 4, 18, 9, 20, 128, 19, 87, 69, 14]
    elif (FLAGS.seed == 1):
        label=  [165, 61, 108, 91, 58, 156, 104, 163, 106, 1]
    elif (FLAGS.seed == 2):
        label=  [199, 105, 120, 101, 149, 182, 124, 152, 111, 53]
    elif (FLAGS.seed == 3):
        label=  [213, 160, 138, 169, 34, 177, 210, 172, 190, 109]
    elif (FLAGS.seed == 4):
#                  1    2    3   4   5    6    7    8    9
        label=  [233, 176, 194, 33, 28, 198, 245, 289, 192, 225]
    elif (FLAGS.seed == 5):
        label=  [199, 226, 138, 266, 343, 215, 326, 318, 216, 219]
    elif (FLAGS.seed == 6):  # Based on prototype refining for seed=2
    #From seeds   3     2    2    2    2    2    2    2    2  1
        label=  [213, 105, 120, 101, 149, 182, 124, 152, 111, 1]
#    elif (FLAGS.seed == 6):  # Based on prototype refining for balance=0
#        label=  [213, 105, 108, 91, 149, 182, 210, 163, 190, 1]
#        label=  [20483, 8867, 47593, 23531, 14445, 35952, 28881, 8210, 30034, 40253]
#    elif (FLAGS.seed == 7):  # Based on prototype refining for balance=2
#        label=  [213, 226, 138, 169, 343, 177, 245, 318, 106, 225]
#        label=  [46819, 19300, 13322, 29007, 44816, 1427, 14931, 344, 31772, 27357]
    elif (FLAGS.seed == 7):  # Based on prototype refining for seed=0
    #From seeds   4    4    4   1    4    2   4    4    4    4
        label=  [233, 176, 194, 91, 28, 124, 245, 289, 192, 225]
    #From seeds   3   0    3  0    2    2   0  0    0   0
#        label=  [213, 4, 138, 9, 149, 182, 19, 87, 69, 14]
    #From seeds    3  0   0  0    5   0    0   0   0   0
#        label=  [213, 4, 18, 9, 343, 128, 19, 87, 69, 14]
    elif (FLAGS.seed == 8):
        label=  [42112, 46875, 27460, 15846, 35142, 45802, 39501, 8628, 48571, 49791]
    elif (FLAGS.seed == 9):
    #From seeds   0  0    3  0   0    2   0  0    0   0
        label=  [35, 4, 138, 9, 20, 182, 19, 87, 69, 14]
    #From seeds   3     3    3   1   3    3    3    3    3  3
#        label=  [213, 160, 138, 91, 34, 177, 210, 172, 190, 109]
    elif (FLAGS.seed == 10):
    #From seeds   3   0    3  0    2    2   0  0    0   0
        label=  [213, 4, 138, 9, 149, 182, 19, 87, 69, 14]
    #From seeds   3     3    1   3   3    3    3    3    3  3
#        label=  [213, 160, 108, 169, 34, 177, 210, 172, 190, 109]
    elif (FLAGS.seed == 11):
    #From seeds   3     3    3   3   2    3    3    3    3  3
        label=  [213, 160, 138, 169, 149, 177, 210, 172, 190, 109]
    elif (FLAGS.seed == 12):
    #From seeds   3     3    3   3   3     2    3    3    3   3
        label=  [213, 160, 138, 169, 34, 182, 210, 172, 190, 109]
    else:
        print(FLAGS.seed, " seed not defined")
        exit(1)
    print("label ",label)
#### End  Prototype code
    try:
        with tf.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i)
                    class_id[i].append(count)
                    if count in label:
                        print("class, count ",i,count)
                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    print("=> id_class.shape,class_id.shape= ", len(id_class) ,len(class_id) )
    nclass = len(class_id)
    for i in range(nclass):
        print("i,class_id[i][0:32]= ",i,class_id[i][0:32])
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
    train_stats /= train_stats.max()
    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] = 1

    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]
    del class_id
    print("===> label= ", label)
    label = frozenset([int(x) for x in label])
    print("===> label= ", label)
    if 'stl10' in argv[1] and FLAGS.size == 1000:
        data = tf.gfile.Open(os.path.join(libml_data.DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
        label = frozenset(list(map(int, data.split('\n')[FLAGS.seed].split())))

    print('Creating split in %s' % target)
    tf.gfile.MakeDirs(os.path.dirname(target))
    with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files:
            for record in tf.python_io.tf_record_iterator(input_file):
                if pos in label:
                    writer_label.write(record)
                pos += 1
                loop.update()
        loop.close()
    with tf.gfile.Open(target + '-label.json', 'w') as writer:
        writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
