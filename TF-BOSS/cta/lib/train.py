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


import tensorflow as tf
import numpy as np
from absl import flags
import statistics 
import os

from fully_supervised.lib.train import ClassifyFullySupervised
from libml import data
from libml.augment import AugmentPoolCTA
from libml.ctaugment import CTAugment
from libml.train import ClassifySemi
from tqdm import trange, tqdm

FLAGS = flags.FLAGS

flags.DEFINE_integer('adepth', 2, 'Augmentation depth.')
flags.DEFINE_float('adecay', 0.99, 'Augmentation decay.')
flags.DEFINE_float('ath', 0.80, 'Augmentation threshold.')


class CTAClassifySemi(ClassifySemi):
    """Semi-supervised classification."""
    AUGMENTER_CLASS = CTAugment
    AUGMENT_POOL_CLASS = AugmentPoolCTA

    @classmethod
    def cta_name(cls):
        return '%s_depth%d_th%.2f_decay%.3f' % (cls.AUGMENTER_CLASS.__name__,
                                                FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.augmenter = self.AUGMENTER_CLASS(FLAGS.adepth, FLAGS.ath, FLAGS.adecay)
        self.best_acc=0
        self.best_accStd=0
        self.counter=0

    def gen_labeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = True
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def gen_unlabeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = False
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def train_step(self, train_session, gen_labeled, gen_unlabeled):
        x, y = gen_labeled(), gen_unlabeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset, name):
            """Return numpy arrays of all the samples from a dataset."""
            pbar = tqdm(desc='Caching %s examples' % name)
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])
                pbar.update()

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            pbar.close()
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test.parse(), name='test')
            self.tmp.cache.valid = collect_samples(self.dataset.valid.parse(), name='valid')
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(10000).parse(),
                                                           name='train_labeled')
            self.tmp.cache.train_original = collect_samples(self.dataset.train_original.parse(),
                                                           name='train_original')

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        class_acc = {}
        best_class_acc = {}
        self.counter += 1
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]
            predicted = []

            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            pred = predicted.argmax(1)
            probs = predicted.max(1)
            accuracies.append((pred == labels).mean() * 100)
#####  New Code to compute class accuracies
            if subset == 'test' and self.counter%8 == 0:
                labls = labels + 1
                acc = []
                for i in range(10):
                    cls = i+1
                    mask = 1*(labls == cls)
                    nsamples = mask.sum()
                    mask = 2*mask - 1
                    acc.append((pred+1 == labls*mask).sum()/nsamples * 100)
                    class_acc[subset] = acc
                print("Class accuracies for Test: ",class_acc)
                print("Class accuracies Mean= {:.2f}, STD= {:.2f} ".format(statistics.mean(class_acc['test']),statistics.stdev(class_acc['test'])) )
                testStd = self.best_acc - statistics.stdev(class_acc['test'])
                if testStd  > self.best_accStd:
                    self.best_accStd = testStd

        testAcc = float(accuracies[2])
        if testAcc  > self.best_acc:
            self.best_acc = testAcc
            images, labels = self.tmp.cache['train_original']
            predicted = []
            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            pred = predicted.argmax(1)
            probs = predicted.max(1)
            top = np.argsort(-probs,axis=0)
            trainAcc = ((pred == labels).mean() * 100)
            print("Accuracy of the unlabeled training data =  ",trainAcc)

            balanceIndx = self.train_dir.find('balance')
            balance = self.train_dir[balanceIndx+7]
            seedIndx = self.train_dir.find('@')
            seed = self.train_dir[seedIndx-1]
            fname="B"+balance+"S"+seed
            
            unique_train_counts = [0]*self.nclass
            nPreds = 1500
            while (nPreds < 50000 and  min(unique_train_counts) < 50):
                nPreds +=500
                unique_train_pseudo_labels, unique_train_counts = np.unique(pred[top[:nPreds]], return_counts=True)
            print("nPreds= ",nPreds, "fname= ",fname, "unique_train_counts= ", unique_train_counts," for classes: ", unique_train_pseudo_labels)

            unique_train_pseudo_labels, unique_train_counts = np.unique(pred, return_counts=True)
            print("Number of training pseudo-labels in each class: ", unique_train_counts," for classes: ", unique_train_pseudo_labels)

            np.save("top_labels"+fname, pred[top[:nPreds]])
            np.save("true_labels"+fname, labels[top[:nPreds]])
            # For some reason this program is not reading in the first training image.  
            # Add 1 to top to align top with the actual training image numbers.
            top = top + 1
            np.save("top_probs"+fname, top[:nPreds])
##### End of new code
        if verbose:
            acc = list([self.tmp.step >> 10] + accuracies)
            acc.append(self.best_acc)
            acc.append(self.best_accStd)
            tup = tuple(acc)
            self.train_print('kimg %-5d  accuracy train/valid/test/best_test/acc-std  %.2f  %.2f  %.2f  %.2f  %.2f' % tup)

#        if verbose:
#            self.train_print('kimg %-5d  accuracy train/valid/test  %.2f  %.2f  %.2f' %
#                             tuple([self.tmp.step >> 10] + accuracies))
#        self.train_print(self.augmenter.stats())
        return np.array(accuracies, 'f')


class CTAClassifyFullySupervised(ClassifyFullySupervised, CTAClassifySemi):
    """Fully-supervised classification."""

    def train_step(self, train_session, gen_labeled):
        x = gen_labeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)
