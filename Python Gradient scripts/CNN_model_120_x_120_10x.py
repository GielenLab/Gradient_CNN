# -*- coding: utf-8 -*-

import tensorflow as tf


def bead_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  
 
  input_layer = tf.compat.v1.reshape(features["x"], [-1, 120, 120, 1], name = 'tf_reshape1')
  
  # Convolutional Layer #1

  conv1 = tf.compat.v1.layers.conv2d(
      inputs=input_layer,
      filters=2, #8
      kernel_size=[15, 15],
      padding="same",
      activation=tf.compat.v1.nn.relu)

 # norm1=tf.layers.batch_normalization(inputs=conv1)
  # Pooling Layer #1
  pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.compat.v1.layers.conv2d(
      inputs=pool1,
      filters=4, #16
      kernel_size=[15, 15],
      padding="same",
      activation=tf.compat.v1.nn.relu)
 # norm2=tf.layers.batch_normalization(inputs=conv2)
    # Pooling Layer #2♣
  pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#  
    # Convolutional Layer #2
  conv3 = tf.compat.v1.layers.conv2d(
      inputs=pool2,
      filters=8, #16
      kernel_size=[15, 15],
      padding="same",
      activation=tf.compat.v1.nn.relu)
 # norm3=tf.layers.batch_normalization(inputs=conv3)
    # Pooling Layer #2
  pool3 = tf.compat.v1.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  
  # Flatten tensor into a batch of vectors
  pool3_flat = tf.compat.v1.reshape(pool3, [-1, 15 * 15 * 8])

  # Dense Layer
  dense = tf.compat.v1.layers.dense(inputs=pool3_flat, units=128, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.compat.v1.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer = Number of classes
  logits = tf.compat.v1.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.compat.v1.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.compat.v1.nn.softmax(logits, name="softmax_tensor")
  }
  
  
  if mode == tf.compat.v1.estimator.ModeKeys.PREDICT:
    return tf.compat.v1.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  accuracy, update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])
  my_acc = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(tf.compat.v1.cast(labels, tf.compat.v1.int64), predictions['classes']), tf.compat.v1.float32))


  # Configure the Training Op (for TRAIN mode)
  if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001) 
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.compat.v1.train.get_global_step())
    logging_hook = tf.compat.v1.train.LoggingTensorHook({"My accuracy": my_acc}, every_n_iter=100)
    return tf.compat.v1.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.compat.v1.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.compat.v1.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  