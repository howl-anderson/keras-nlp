# Copyright 2022 The KerasNLP Authors
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

from keras_nlp.augmention.base_augmenter import BaseAugmenter
from keras_nlp.layers.random_deletion import (
    RandomDeletion as RandomDeletionLayer,
)


class RandomDeletion(BaseAugmenter):
    """An Augmenter delete tokens randomly.

    This augmenter implemented the deletion
    augmentation as described in the paper [EDA: Easy Data Augmentation
    Techniques for Boosting Performance on Text Classification Tasks]
    (https://arxiv.org/pdf/1901.11196.pdf).

    Args:
        **kwargs: see `keras_nlp.layers.random_deletion.RandomDeletion` for details

    Examples:

    Combining with tf.data.DataSet.
    >>> keras.utils.set_random_seed(1337)
    >>> dataset = tf.data.Dataset.from_tensor_slices(
    ...      tf.strings.split(
    ...          [
    ...              "Knowledge is power",
    ...              "The quick brown fox jumps over the lazy dog",
    ...          ]
    ...      )
    ...  )
    >>> new_ds = dataset.flat_map(
    ...      RandomDeletion(rate=0.4, max_deletions=1, seed=42).flat_map_fn
    ...  )
    >>> for sentence in new_ds:
    ...      sentence_str = tf.strings.reduce_join(
    ...          sentence, separator=" ", axis=-1
    ...      )
    ...      print(sentence_str)
    tf.Tensor(b'is power', shape=(), dtype=string)
    tf.Tensor(b'quick brown fox jumps over the lazy dog', shape=(), dtype=string)

    Reference:
     - [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/pdf/1901.11196.pdf)
    """

    def __init__(self, **kwargs):
        self.layer_instance = RandomDeletionLayer(**kwargs)

    def flat_map_fn(self, inputs) -> tf.data.Dataset:
        result = self.layer_instance(inputs)
        return tf.data.Dataset.from_tensors(result)
