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
from tensorflow import keras

from keras_nlp.augmention.random_deletion import RandomDeletion


class RandomDeletionTest(tf.test.TestCase):
    def test_dataset_function(self):
        keras.utils.set_random_seed(1337)

        dataset = tf.data.Dataset.from_tensor_slices(
            tf.strings.split(
                [
                    "Knowledge is power",
                    "The quick brown fox jumps over the lazy dog",
                ]
            )
        )
        new_ds = dataset.flat_map(
            RandomDeletion(rate=0.4, max_deletions=1, seed=42).flat_map_fn
        )

        expected_results = [
            b"is power",
            b"quick brown fox jumps over the lazy dog",
        ]
        for element, expected_result in zip(new_ds, expected_results):
            real_result = tf.strings.reduce_join(
                element, separator=" ", axis=-1
            )
            self.assertAllEqual(real_result, expected_result)
