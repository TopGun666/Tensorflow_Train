import tensorflow as tf
import tensorflow.contrib.slim as slim
import math

class TriangulationEmbeddingModule:

# End to end trainable Triangulation Embedding module
    def __init__(self,
                 feature_size,
                 num_descriptors,
                 num_anchors,
                 add_batch_norm,
                 is_training,
                 scope_id = None):

        ''' Initialize class TriangulationEmbeddingModule.
        :param feature_size: int
            The number of features in each descriptors.
        :param num_descriptors: int
            Total number of descriptors.
        :param num_anchors: int
            The number of anchors.
        :param add_batch_norm: bool
            True iff batch normalization is used at the end.
        :param is_training: bool
            True iff the module is being trained.
        :param scope_id: String
            The scope id.
        '''

        self.feature_size = feature_size
        self.num_descriptors = num_descriptors
        self.num_anchors = num_anchors
        self.add_batch_norm = add_batch_norm
        self.is_training = is_training
        self.scope_id = scope_id

    def forward(self, inputs, **unused_params):
        '''Forward method for TriangulationEmbeddingModule.
        :param self:
        :param input:2D Tensor with dimension '(batch_size * num_descriptors) * feature_size'
        :param unused_params:3D Tensor with dimension 'batch_size * num_descriptors * (feature_size * num_anchors)'
        :return:
        '''
        anchors_weights = tf.get_variable("anchor_weight{}".format("" if self.scope_id is None else str(self.scope_id))
                                          [self.feature_size, self.num_anchors],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.num_anchors)),
                                          dtype=tf.float32)
        tf.summary.histogram("anchor_weights{}".format("" if self.scope_id is None else str(self.scope_id)),
                             anchors_weights)

        anchors_weights = tf.transpose(anchors_weights)
        anchors_weights = tf.reshape(anchors_weights, [1, self.feature_size * self.num_anchors])

        # Tile inputs to subtract all anchors.
        tiled_inputs = tf.tile(inputs, [1, self.num_anchors])
        # ->(batch_size * num_descriptors) x (feature_size * num_anchors)
        t_emb = tf.reshape(t_emb, [-1, self.num_anchors, self.feature_size])
        # -> (batch_size * num_descriptors) x num_anchors x feature_size; Keep normalized residuals.
        t_emb = tf.nn.l2_normalize(t_emb, 2)
        t_emb = tf.reshape(t_emb, [-1, self.num_descriptors, self.feature_size * self.num_anchors])
        # -> (batch_size * num_descriptors) x (feature_size * num_anchors)

        if self.add_batch_norm:
            t_emb = slim.batch_norm(
                t_emb,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="t_emb_bn")

        activation = tf.reshape(t_emb, [-1, self.num_descriptors, self.feature_size * self.num_anchors])

        return activation

    class TemporalDifferenceDescriptors:
# Return the difference between consecutive descriptors to extract temporal information
        def __init__(self,
                     num_descriptors,
                     num_anchors,
                     add_batch_norm,
                     is_training,
                     scope_id=None):
            '''Initialize class TemporalDifferenceDescriptors