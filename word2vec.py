"""
阅读tensorflow word2vec的一些源代码，下面是阅读顺序
=====================================================
1. tf.nn.nce_loss
nce的源码分成两个部分，首先是样本采样，而后使用交叉熵计算loss，
关于_compute_sampled_logits的源码见第二部分
关于sigmoid_cross_entropy_with_logits的源码见第三部分
"""
def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
  """Computes and returns the noise-contrastive estimation training loss.
  See [Noise-contrastive estimation: A new estimation principle for
  unnormalized statistical
  models](http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
  Also see our [Candidate Sampling Algorithms
  Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)
  A common use case is to use this method for training, and calculate the full
  sigmoid loss for evaluation or inference. In this case, you must set
  `partition_strategy="div"` for the two losses to be consistent, as in the
  following example:
  ```python
  if mode == "train":
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        ...,
        partition_strategy="div")
  elif mode == "eval":
    logits = tf.matmul(inputs, tf.transpose(weights))
    logits = tf.nn.bias_add(logits, biases)
    labels_one_hot = tf.one_hot(labels, n_classes)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels_one_hot,
        logits=logits)
    loss = tf.reduce_sum(loss, axis=1)
  ```
  Note: By default this uses a log-uniform (Zipfian) distribution for sampling,
  so your labels must be sorted in order of decreasing frequency to achieve
  good results.  For more details, see
  `tf.random.log_uniform_candidate_sampler`.
  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.
  Note: It would be useful to allow a variable number of target classes per
  example.  We hope to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.
  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of negative classes to randomly sample
        per batch. This single sample of negative classes is evaluated for each
        element in the batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  If set to
        `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
        learning to generate log-odds instead of log probabilities.  See
        our [Candidate Sampling Algorithms Reference]
        (https://www.tensorflow.org/extras/candidate_sampling.pdf).
        Default is False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.
  """
  logits, labels = _compute_sampled_logits(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name)
  sampled_losses = sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name="sampled_losses")
  # sampled_losses is batch_size x {true_loss, sampled_losses...}
  # We sum out true and sampled losses.
  return _sum_rows(sampled_losses)
"""
2. _compute_sampled_logits源码解析
"""
def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            seed=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.
  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).
  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.
  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The (possibly-partitioned)
        class biases.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
    seed: random seed for candidate sampling. Default to None, which doesn't set
        the op-level random seed for candidate sampling.
  Returns:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    out_labels: A Tensor object with the same shape as `out_logits`.
  """

  if isinstance(weights, variables.PartitionedVariable):
    weights = list(weights)
  if not isinstance(weights, list):
    weights = [weights]

  with ops.name_scope(name, "compute_sampled_logits",
                      weights + [biases, inputs, labels]):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = (
        array_ops.stop_gradient(s) for s in sampled_values)
    # pylint: enable=unpacking-non-sequence
    sampled = math_ops.cast(sampled, dtypes.int64)

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat([labels_flat, sampled], 0)

    # Retrieve the true weights and the logits of the sampled weights.

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    if all_w.dtype != inputs.dtype:
      all_w = math_ops.cast(all_w, inputs.dtype)

    # true_w shape is [batch_size * num_true, dim]
    true_w = array_ops.slice(all_w, [0, 0],
                             array_ops.stack(
                                 [array_ops.shape(labels_flat)[0], -1]))

    sampled_w = array_ops.slice(
        all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # Apply X*W', which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    # Retrieve the true and sampled biases, compute the true logits, and
    # add the biases to the true and sampled logits.
    all_b = embedding_ops.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
    if all_b.dtype != inputs.dtype:
      all_b = math_ops.cast(all_b, inputs.dtype)
    # true_b is a [batch_size * num_true] tensor
    # sampled_b is a [num_sampled] float tensor
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(
          math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                        "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = array_ops.concat(
          [array_ops.shape(labels)[:1],
           array_ops.expand_dims(num_sampled, 0)], 0)
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += gen_sparse_ops.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)

    # true_logits is a float tensor, ones_like(true_logits) is a float
    # tensor of ones. We then divide by num_true to ensure the per-example
    # labels sum to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_labels

"""
4. tensorflow.python.ops.candidate_sampling_ops.fixed_unigram_candidate_sampler:
可以自定义采样分布，其中distortion参数设为0.75即为word2vec原论文中的采样函数，vocab_file
为读取每一个词频字典的文件位置。实际实现word2vec可以使用unigrams参数，此参数为一个列表，且必须
与词编码的序列相同（ps：我终于找到了负采样的实现了！）
"""
def fixed_unigram_candidate_sampler(true_classes,
                                    num_true,
                                    num_sampled,
                                    unique,
                                    range_max,
                                    vocab_file='',
                                    distortion=1.0,
                                    num_reserved_ids=0,
                                    num_shards=1,
                                    shard=0,
                                    unigrams=(),
                                    seed=None,
                                    name=None):
  """Samples a set of classes using the provided (fixed) base distribution.
  This operation randomly samples a tensor of sampled classes
  (`sampled_candidates`) from the range of integers `[0, range_max)`.
  The elements of `sampled_candidates` are drawn without replacement
  (if `unique=True`) or with replacement (if `unique=False`) from
  the base distribution.
  The base distribution is read from a file or passed in as an
  in-memory array. There is also an option to skew the distribution by
  applying a distortion power to the weights.
  In addition, this operation returns tensors `true_expected_count`
  and `sampled_expected_count` representing the number of times each
  of the target classes (`true_classes`) and the sampled
  classes (`sampled_candidates`) is expected to occur in an average
  tensor of sampled classes.  These values correspond to `Q(y|x)`
  defined in [this
  document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
  If `unique=True`, then these are post-rejection probabilities and we
  compute them approximately.
  Args:
    true_classes: A `Tensor` of type `int64` and shape `[batch_size,
      num_true]`. The target classes.
    num_true: An `int`.  The number of target classes per training example.
    num_sampled: An `int`.  The number of classes to randomly sample.
    unique: A `bool`. Determines whether all sampled classes in a batch are
      unique.
    range_max: An `int`. The number of possible classes.
    vocab_file: Each valid line in this file (which should have a CSV-like
      format) corresponds to a valid word ID. IDs are in sequential order,
      starting from num_reserved_ids. The last entry in each line is expected
      to be a value corresponding to the count or relative probability. Exactly
      one of `vocab_file` and `unigrams` needs to be passed to this operation.
    distortion: The distortion is used to skew the unigram probability
      distribution.  Each weight is first raised to the distortion's power
      before adding to the internal unigram distribution. As a result,
      `distortion = 1.0` gives regular unigram sampling (as defined by the vocab
      file), and `distortion = 0.0` gives a uniform distribution.
    num_reserved_ids: Optionally some reserved IDs can be added in the range
      `[0, num_reserved_ids)` by the users. One use case is that a special
      unknown word token is used as ID 0. These IDs will have a sampling
      probability of 0.
    num_shards: A sampler can be used to sample from a subset of the original
      range in order to speed up the whole computation through parallelism. This
      parameter (together with `shard`) indicates the number of partitions that
      are being used in the overall computation.
    shard: A sampler can be used to sample from a subset of the original range
      in order to speed up the whole computation through parallelism. This
      parameter (together with `num_shards`) indicates the particular partition
      number of the operation, when partitioning is being used.
    unigrams: A list of unigram counts or probabilities, one per ID in
      sequential order. Exactly one of `vocab_file` and `unigrams` should be
      passed to this operation.
    seed: An `int`. An operation-specific seed. Default is 0.
    name: A name for the operation (optional).
  Returns:
    sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
      The sampled classes.
    true_expected_count: A tensor of type `float`.  Same shape as
      `true_classes`. The expected counts under the sampling distribution
      of each of `true_classes`.
    sampled_expected_count: A tensor of type `float`. Same shape as
      `sampled_candidates`. The expected counts under the sampling distribution
      of each of `sampled_candidates`.
  """
  seed1, seed2 = random_seed.get_seed(seed)
  return gen_candidate_sampling_ops.fixed_unigram_candidate_sampler(
      true_classes, num_true, num_sampled, unique, range_max,
      vocab_file=vocab_file, distortion=distortion,
      num_reserved_ids=num_reserved_ids, num_shards=num_shards, shard=shard,
      unigrams=unigrams, seed=seed1, seed2=seed2, name=name)
