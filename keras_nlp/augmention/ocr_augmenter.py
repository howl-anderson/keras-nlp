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

try:
    import nlpaug.augmenter.char as nac
except ImportError:
    nac = None


class OcrAugmenter(BaseAugmenter):
    """An Augmenter substitute character by pre-defined OCR error.

    This augmenter simulates ocr errors with random values. For example, OCR may incorrectly identify an I as a 1 and a 0 as an o.

    Args:
        **kwargs: see `nlpaug.augmenter.char.ocr.OcrAug` for details

    Examples:

    Combining with tf.data.DataSet.
    >>> keras.utils.set_random_seed(1337)
    >>> dataset = tf.data.Dataset.from_tensor_slices(
    ...     tf.strings.split(
    ...         [
    ...             "Knowledge is power",
    ...             "The quick brown fox jumps over the lazy dog",
    ...         ]
    ...     )
    ... )
    >>> new_ds = dataset.flat_map(OcrAugmenter().flat_map_fn)
    >>> for sentence in new_ds:
    ...      sentence_str = tf.strings.reduce_join(
    ...          sentence, separator=" ", axis=-1
    ...      )
    ...      print(sentence_str)
    tf.Tensor(b'Knowledge is p0wek', shape=(), dtype=string)
    tf.Tensor(b'The quick brown f0x jomp8 0vek the lazy dog', shape=(), dtype=string)
    """

    def __init__(self, **kwargs):
        if nac is None:
            raise ImportError(
                "OcrAugmenter requires the `nlpaug` package."
                "Please install it with `pip install nlpaug`."
            )

        self.n = kwargs.pop("n", 1)
        self.num_thread = kwargs.pop("num_thread", 1)

        self.layer_instance = nac.OcrAug(**kwargs)

    def _py_augment(self, inputs):
        text_str = (
            tf.strings.reduce_join(inputs, separator=" ", axis=-1)
            .numpy()
            .decode()
        )
        augmented_text_str = self.layer_instance.augment(
            text_str, n=self.n, num_thread=self.num_thread
        )
        return tf.strings.split(augmented_text_str)

    def _augment(self, inputs):
        return tf.py_function(
            self._py_augment,
            [inputs],
            tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.string),
        )

    def flat_map_fn(self, inputs) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(self._augment(inputs))
