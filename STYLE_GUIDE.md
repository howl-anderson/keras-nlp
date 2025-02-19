# Style Guide

## Use `black`

For the most part, following our code style is very simple, we just use
[black](https://github.com/psf/black) to format code. See our
[Contributing Guide](CONTRIBUTING.md) for how to run our formatting scripts.

## Import keras and keras_nlp as top-level objects

Prefer importing `tf`, `keras` and `keras_nlp` as top-level objects. We want
it to be clear to a reader which symbols are from `keras_nlp` and which are
from core `keras`.

For guides and examples using KerasNLP, the import block should look as follows:

```python
import keras_nlp
import tensorflow as tf
from tensorflow import keras
```

❌ `tf.keras.activations.X`<br/>
✅ `keras.activations.X`

❌ `layers.X`<br/>
✅ `keras.layers.X` or `keras_nlp.layers.X`

❌ `Dense(1, activation='softmax')`<br/>
✅ `keras.layers.Dense(1, activation='softmax')`

For KerasNLP library code, `keras_nlp` will not be directly imported, but
`keras` should still be used as a top-level object used to access library
symbols.

## Ideal layer style

When writing a new KerasNLP layer (or tokenizer or metric), please make sure to
do the following:

- Accept `**kwargs` in `__init__` and forward this to the super class.
- Keep a python attribute on the layer for each `__init__` argument to the
  layer. The name and value should match the passed value.
- Write a `get_config()` which chains to super.
- Document the layer behavior thoroughly including call behavior though a
  class level docstring. Generally methods like `build()` and `call()` should
  not have their own docstring.
- Document the
  [masking](https://keras.io/guides/understanding_masking_and_padding/) behavior
  of the layer in the class level docstring as well.
- Always include usage examples using the full symbol location in `keras_nlp`.
- Include a reference citation if applicable.

````python
class PositionEmbedding(keras.layers.Layer):
    """A layer which learns a position embedding for input sequences.

    This class accepts a single dense tensor as input, and will output a
    learned position embedding of the same shape.

    This class assumes that in the input tensor, the last dimension corresponds
    to the features, and the dimension before the last corresponds to the
    sequence.

    This layer does not supporting masking, but can be combined with a
    `keras.layers.Embedding` for padding mask support.

    Args:
        sequence_length: The maximum length of the dynamic sequence.

    Examples:

    Direct call.
    >>> layer = keras_nlp.layers.PositionEmbedding(sequence_length=10)
    >>> layer(tf.zeros((8, 10, 16))).shape
    TensorShape([8, 10, 16])

    Combining with a token embedding.
    ```python
    seq_length = 50
    vocab_size = 5000
    embed_dim = 128
    inputs = keras.Input(shape=(seq_length,))
    token_embeddings = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    position_embeddings = keras_nlp.layers.PositionEmbedding(
        sequence_length=seq_length
    )(token_embeddings)
    outputs = token_embeddings + position_embeddings
    ```

    Reference:
     - [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        sequence_length,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)

    def build(self, input_shape):
        super().build(input_shape)
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            "embeddings",
            shape=[self.sequence_length, feature_size],
        )

    def call(self, inputs):
        shape = tf.shape(inputs)
        input_length = shape[-2]
        position_embeddings = self.position_embeddings[:input_length, :]
        return tf.broadcast_to(position_embeddings, shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
            }
        )
        return config
````
