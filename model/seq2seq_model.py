# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import copy
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss


# from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import _extract_argmax_and_embed
class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
          dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                if forward_only:
                    return tf.zeros(batch_size, tf.float32)
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        weights=local_w_t,
                        biases=local_b,
                        labels=labels,
                        inputs=local_inputs,
                        num_sampled=num_samples,
                        num_classes=self.target_vocab_size),
                    dtype)

            softmax_loss_function = sampled_loss

        def model_with_buckets(encoder_inputs,
                               decoder_inputs,
                               targets,
                               weights,
                               buckets,
                               seq2seq,
                               softmax_loss_function=None,
                               per_example_loss=False,
                               name=None):
            """Create a sequence-to-sequence model with support for bucketing.
            The seq2seq argument is a function that defines a sequence-to-sequence model,
            e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
                x, y, rnn_cell.GRUCell(24))
            Args:
              encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
              decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
              targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
              weights: List of 1D batch-sized float-Tensors to weight the targets.
              buckets: A list of pairs of (input size, output size) for each bucket.
              seq2seq: A sequence-to-sequence model function; it takes 2 input that
                agree with encoder_inputs and decoder_inputs, and returns a pair
                consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
              softmax_loss_function: Function (labels, logits) -> loss-batch
                to be used instead of the standard softmax (the default if this is None).
                **Note that to avoid confusion, it is required for the function to accept
                named arguments.**
              per_example_loss: Boolean. If set, the returned loss will be a batch-sized
                tensor of losses for each sequence in the batch. If unset, it will be
                a scalar with the averaged loss from all examples.
              name: Optional name for this operation, defaults to "model_with_buckets".
            Returns:
              A tuple of the form (outputs, losses), where:
                outputs: The outputs for each bucket. Its j'th element consists of a list
                  of 2D Tensors. The shape of output tensors can be either
                  [batch_size x output_size] or [batch_size x num_decoder_symbols]
                  depending on the seq2seq model used.
                losses: List of scalar Tensors, representing losses for each bucket, or,
                  if per_example_loss is set, a list of 1D batch-sized float Tensors.
            Raises:
              ValueError: If length of encoder_inputs, targets, or weights is smaller
                than the largest (last) bucket.
            """
            if len(encoder_inputs[0]) < buckets[-1][0]:
                raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                                 "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
            if len(targets) < buckets[-1][1]:
                raise ValueError("Length of targets (%d) must be at least that of last"
                                 "bucket (%d)." % (len(targets), buckets[-1][1]))
            if len(weights) < buckets[-1][1]:
                raise ValueError("Length of weights (%d) must be at least that of last"
                                 "bucket (%d)." % (len(weights), buckets[-1][1]))

            all_inputs = encoder_inputs + decoder_inputs + targets + weights
            losses = []
            outputs = []
            states = []
            with ops.name_scope(name, "model_with_buckets", all_inputs):
                for j, bucket in enumerate(buckets):
                    with variable_scope.variable_scope(
                            variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                        encoder_inputss = []
                        for x in encoder_inputs:
                            encoder_inputss.append(x[:bucket[0]])
                        bucket_outputs, bucket_states = seq2seq(encoder_inputss,
                                                                decoder_inputs[:bucket[1]])
                        outputs.append(bucket_outputs)
                        states.append(bucket_states)
                        if per_example_loss:
                            losses.append(
                                sequence_loss_by_example(
                                    outputs[-1],
                                    targets[:bucket[1]],
                                    weights[:bucket[1]],
                                    softmax_loss_function=softmax_loss_function))
                        else:
                            losses.append(
                                sequence_loss(
                                    outputs[-1],
                                    targets[:bucket[1]],
                                    weights[:bucket[1]],
                                    softmax_loss_function=softmax_loss_function))

            return outputs, losses, states
        #static rnn
        '''
        def static_rnn(cell,
                       inputs,
                       initial_state=None,
                       dtype=None,
                       sequence_length=None,
                       scope=None):
            """Creates a recurrent neural network specified by RNNCell `cell`.
            The simplest form of RNN network generated is:
            ```python
              state = cell.zero_state(...)
              outputs = []
              for input_ in inputs:
                output, state = cell(input_, state)
                outputs.append(output)
              return (outputs, state)
            ```
            However, a few other options are available:
            An initial state can be provided.
            If the sequence_length vector is provided, dynamic calculation is performed.
            This method of calculation does not compute the RNN steps past the maximum
            sequence length of the minibatch (thus saving computational time),
            and properly propagates the state at an example's sequence length
            to the final state output.
            The dynamic calculation performed is, at time `t` for batch row `b`,
            ```python
              (output, state)(b, t) =
                (t >= sequence_length(b))
                  ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
                  : cell(input(b, t), state(b, t - 1))
            ```
            Args:
              cell: An instance of RNNCell.
              inputs: A length T list of inputs, each a `Tensor` of shape
                `[batch_size, input_size]`, or a nested tuple of such elements.
              initial_state: (optional) An initial state for the RNN.
                If `cell.state_size` is an integer, this must be
                a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
                If `cell.state_size` is a tuple, this should be a tuple of
                tensors having shapes `[batch_size, s] for s in cell.state_size`.
              dtype: (optional) The data type for the initial state and expected output.
                Required if initial_state is not provided or RNN state has a heterogeneous
                dtype.
              sequence_length: Specifies the length of each sequence in inputs.
                An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
              scope: VariableScope for the created subgraph; defaults to "rnn".
            Returns:
              A pair (outputs, state) where:
              - outputs is a length T list of outputs (one for each input), or a nested
                tuple of such elements.
              - state is the final state
            Raises:
              TypeError: If `cell` is not an instance of RNNCell.
              ValueError: If `inputs` is `None` or an empty list, or if the input depth
                (column size) cannot be inferred from inputs via shape inference.
            """

            if not _like_rnncell(cell):
                raise TypeError("cell must be an instance of RNNCell")
            if not nest.is_sequence(inputs):
                raise TypeError("inputs must be a sequence")
            if not inputs:
                raise ValueError("inputs must not be empty")

            outputs = []
            # Create a new scope in which the caching device is either
            # determined by the parent scope, or is set to place the cached
            # Variable using the same placement as for the rest of the RNN.
            with vs.variable_scope(scope or "rnn") as varscope:
                if varscope.caching_device is None:
                    varscope.set_caching_device(lambda op: op.device)

                # Obtain the first sequence of the input
                first_input = inputs
                while nest.is_sequence(first_input):
                    first_input = first_input[0]

                # Temporarily avoid EmbeddingWrapper and seq2seq badness
                # TODO(lukaszkaiser): remove EmbeddingWrapper
                if first_input.get_shape().ndims != 1:

                    input_shape = first_input.get_shape().with_rank_at_least(2)
                    fixed_batch_size = input_shape[0]

                    flat_inputs = nest.flatten(inputs)
                    for flat_input in flat_inputs:
                        input_shape = flat_input.get_shape().with_rank_at_least(2)
                        batch_size, input_size = input_shape[0], input_shape[1:]
                        fixed_batch_size.merge_with(batch_size)
                        for i, size in enumerate(input_size):
                            if size.value is None:
                                raise ValueError(
                                    "Input size (dimension %d of inputs) must be accessible via "
                                    "shape inference, but saw value None." % i)
                else:
                    fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

                if fixed_batch_size.value:
                    batch_size = fixed_batch_size.value
                else:
                    batch_size = array_ops.shape(first_input)[0]
                if initial_state is not None:
                    state = initial_state
                else:
                    if not dtype:
                        raise ValueError("If no initial_state is provided, "
                                         "dtype must be specified")
                    state = cell.zero_state(batch_size, dtype)

                if sequence_length is not None:  # Prepare variables
                    sequence_length = ops.convert_to_tensor(
                        sequence_length, name="sequence_length")
                    if sequence_length.get_shape().ndims not in (None, 1):
                        raise ValueError(
                            "sequence_length must be a vector of length batch_size")

                    def _create_zero_output(output_size):
                        # convert int to TensorShape if necessary
                        size = _concat(batch_size, output_size)
                        output = array_ops.zeros(
                            array_ops.stack(size), _infer_state_dtype(dtype, state))
                        shape = _concat(fixed_batch_size.value, output_size, static=True)
                        output.set_shape(tensor_shape.TensorShape(shape))
                        return output

                    output_size = cell.output_size
                    flat_output_size = nest.flatten(output_size)
                    flat_zero_output = tuple(
                        _create_zero_output(size) for size in flat_output_size)
                    zero_output = nest.pack_sequence_as(
                        structure=output_size, flat_sequence=flat_zero_output)

                    sequence_length = math_ops.to_int32(sequence_length)
                    min_sequence_length = math_ops.reduce_min(sequence_length)
                    max_sequence_length = math_ops.reduce_max(sequence_length)

                for time, input_ in enumerate(inputs):
                    if time > 0:
                        varscope.reuse_variables()
                    # pylint: disable=cell-var-from-loop
                    call_cell = lambda: cell(input_, state)
                    # pylint: enable=cell-var-from-loop
                    if sequence_length is not None:
                        (output, state) = _rnn_step(
                            time=time,
                            sequence_length=sequence_length,
                            min_sequence_length=min_sequence_length,
                            max_sequence_length=max_sequence_length,
                            zero_output=zero_output,
                            state=state,
                            call_cell=call_cell,
                            state_size=cell.state_size)
                    else:
                        (output, state) = call_cell()

                    outputs.append(output)

                return (outputs, state)

        # Create the internal multi-layer cell for our RNN.
        '''
        def _extract_argmax_and_embed(embedding,
                                      output_projection=None,
                                      update_embedding=True,

                                      ):
            """Get a loop_function that extracts the previous symbol and embeds it.
            Args:
              embedding: embedding tensor for symbols.
              output_projection: None or a pair (W, B). If provided, each fed previous
                output will first be multiplied by W and added B.
              update_embedding: Boolean; if False, the gradients will not propagate
                through the embeddings.
            Returns:
              A loop function.
            """

            def loop_function(prev, _):
                if output_projection is not None:
                    prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
                prev_symbol = math_ops.argmax(prev, 1)
                # Note that gradients will not propagate through the second parameter of
                # embedding_lookup.
                emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
                if not update_embedding:
                    emb_prev = array_ops.stop_gradient(emb_prev)
                return emb_prev

            return loop_function

        def single_cell():
            return tf.contrib.rnn.GRUCell(size)

        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(size)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            tmp_cell = copy.deepcopy(cell)
            with variable_scope.variable_scope("embedding_rnn_seq2seq") as scope:
                scope.set_dtype(tf.float32)

                # Encoder.
                encoder_cell = copy.deepcopy(cell)
                encoder_cell = core_rnn_cell.EmbeddingWrapper(
                    encoder_cell,
                    embedding_classes=target_vocab_size,
                    embedding_size=size)
                state_input = []
                for encoder_input in encoder_inputs:
                    _, encoder_state = rnn.static_rnn(encoder_cell, encoder_input, dtype=dtype)
                    state_input.append(encoder_state)
                #add state
                state_cell = copy.deepcopy(cell)
                _,final_state = rnn.static_rnn(state_cell, state_input, dtype=dtype)

                # Decoder.
                decoder_cell = copy.deepcopy(cell)
                if output_projection is None:
                    decoder_cell = core_rnn_cell.OutputProjectionWrapper(decoder_cell, target_vocab_size)

                if isinstance(do_decode, bool):
                    initial_state = final_state#encoder_state
                    num_symbols = target_vocab_size
                    embedding_size = size
                    feed_previous = do_decode
                    with variable_scope.variable_scope("embedding_rnn_decoder") as scope:
                        if output_projection is not None:
                            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
                            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
                            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
                            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

                        embedding = variable_scope.get_variable("embedding",
                                                                [num_symbols, embedding_size])
                        loop_function = _extract_argmax_and_embed(
                            embedding, output_projection,
                            True) if feed_previous else None
                        emb_inp = (embedding_ops.embedding_lookup(embedding, i)
                                   for i in decoder_inputs)
                        with variable_scope.variable_scope("rnn_decoder"):
                            state = initial_state
                            decoder_inputs = emb_inp
                            outputs = []
                            prev = None
                            for i, inp in enumerate(decoder_inputs):
                                if loop_function is not None and prev is not None:
                                    with variable_scope.variable_scope("loop_function", reuse=True):
                                        inp = loop_function(prev, i)
                                if i > 0:
                                    variable_scope.get_variable_scope().reuse_variables()
                                output, state = decoder_cell(inp, state)
                                outputs.append(output)
                                if loop_function is not None:
                                    prev = output

                        return outputs, initial_state

                return None

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]*3):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
        #get mul_layer input
        one_layer = buckets[-1][0]
        self.encoder_inputss=[]
        for i in range(3):
            self.encoder_inputss.append(self.encoder_inputs[i*one_layer:(i+1)*one_layer])
        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses, self.states = model_with_buckets(
                self.encoder_inputss, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [
                            tf.matmul(output, output_projection[0]) + output_projection[1]
                            for output in self.outputs[b]
                        ]
        else:
            self.outputs, self.losses, self.states = model_with_buckets(
                self.encoder_inputss, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        one_layer = self.buckets[-1][0]
        # print(encoder_size,decoder_size)
        if len(encoder_inputs[0]) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for i in range(len(encoder_inputs)):
            for l in xrange(len(encoder_inputs[i])):
                input_feed[self.encoder_inputs[i*one_layer+l].name] = encoder_inputs[i][l]
        '''
        for l in xrange(encoder_size):
            # print(type(encoder_inputs[l]),encoder_inputs[l])
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        '''
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
            # for l in xrange(decoder_size):  # Output logits.
            # output_feed.append(self.outputs[bucket_id][l])
        else:
            output_feed = [self.losses[bucket_id],
                           self.states[bucket_id]]
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            '''
            for i in range(len(self.outputs)):
                for l in xrange(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[i][bucket_id][l])
            '''
        # print('hello')
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # outputs[3:]  # Gradient norm, loss, no outputs.
        else:
            return outputs[1], outputs[0], outputs[2:]  # No gradient norm, loss, outputs.

    def step_one(self, session, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        # print(encoder_size,decoder_size)
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            # print(type(encoder_inputs[l]),encoder_inputs[l])
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
            # for l in xrange(decoder_size):  # Output logits.
            # output_feed.append(self.outputs[bucket_id][l])
        else:
            output_feed = [self.losses[bucket_id],
                           self.states[bucket_id]
                           ]  # Loss for this batch.

            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            '''
            for i in range(len(self.outputs)):
                for l in xrange(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[i][bucket_id][l])
            '''
        # print('hello')
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # outputs[3:]  # Gradient norm, loss, no outputs.
        else:
            return outputs[1], outputs[0], outputs[2:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputss, decoder_inputs = [], []
        for i in range(3):
            encoder_inputss.append([])
        bleu_answer = []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_inputs, decoder_input = random.choice(data[bucket_id])


            bleu_answer.append(decoder_input)
            # Encoder inputs are padded and then reversed.
            for i in range(len(encoder_inputs)):
                encoder_input=encoder_inputs[i]
                encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
                '''
                encoder_inputns = []
                encoder_inputns.append(list(reversed(encoder_input + encoder_pad)))
                '''
                encoder_inputss[i].append(list(reversed(encoder_input + encoder_pad)))

            '''
            for encoder_input in encoder_inputs:
                encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
                encoder_inputs =[]
                encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
                encoder_inputss.append(encoder_inputs)
            '''

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputss,batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for encoder_inputs in encoder_inputss:
            batch_encoder_inputs = []
            #print(len(encoder_inputs),self.batch_size)
            #print(len(encoder_inputs[0]),len(encoder_inputs[0][0]), encoder_size)
            for length_idx in xrange(encoder_size):
                batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            batch_encoder_inputss.append(batch_encoder_inputs)



        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.

                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputss, batch_decoder_inputs, batch_weights, bleu_answer
