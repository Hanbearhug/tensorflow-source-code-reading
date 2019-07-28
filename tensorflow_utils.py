"""
本文记录tensorflow的一些小工具的源码解读下面列出已阅读的源码
====================================================================
1.array_ops.slice：在指定起点（begin）截取指定大小的张量，size指在每一个维度上
分别截取的长度。
2. array_ops.stop_gradient：源码未找到，用于停止对于变量input的梯度计算。
3. array_ops.stack：在指定维度上合并张量。源码会自动进行格式转换。
4. tf.losses.sparse_softmax_cross_entropy、tf.losses.softmax_cross_entropys：
使用tf.nn.sparse_softmax_cross_entropy_with_logits、tf.nn.softmax_cross_entropy_with_logits
实现，区别在于，sparse_...loss参数填入时labels参数为[batch_size]，而soft_...loss
参数填入时labels的规格为[batch_size, num_classes]，而在sparse_...loss中在判断labels的
规格时会参考logits的规格。
"""
def slice(input_, begin, size, name=None):
  # pylint: disable=redefined-builtin
  """Extracts a slice from a tensor.
  This operation extracts a slice of size `size` from a tensor `input` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of `input` that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input`. In other
  words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
  want to slice from.
  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:
  `size[i] = input.dim_size(i) - begin[i]`
  This operation requires that:
  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`
  For example:
  ```python
  # 'input' is [[[1, 1, 1], [2, 2, 2]],
  #             [[3, 3, 3], [4, 4, 4]],
  #             [[5, 5, 5], [6, 6, 6]]]
  tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
  tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                              [4, 4, 4]]]
  tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                             [[5, 5, 5]]]
  ```
  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    size: An `int32` or `int64` `Tensor`.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` the same type as `input`.
  """
  return gen_array_ops._slice(input_, begin, size, name=name)

def stack(values, axis=0, name="stack"):
  """Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`;
  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.
  For example:
  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
  tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
  ```
  This is the opposite of unstack.  The numpy equivalent is
  ```python
  tf.stack([x, y, z]) = np.stack([x, y, z])
  ```
  Args:
    values: A list of `Tensor` objects with the same shape and type.
    axis: An `int`. The axis to stack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
    name: A name for this operation (optional).
  Returns:
    output: A stacked `Tensor` with the same type as `values`.
  Raises:
    ValueError: If `axis` is out of the range [-(R+1), R+1).
  """
  if axis == 0:
    try:
      # If the input is a constant list, it can be converted to a constant op
      return ops.convert_to_tensor(values, name=name)
    except (TypeError, ValueError):
      pass  # Input list contains non-constant tensors

  value_shape = ops.convert_to_tensor(values[0], name=name)._shape_tuple()  # pylint: disable=protected-access
  if value_shape is not None:
    expanded_num_dims = len(value_shape) + 1
    if axis < -expanded_num_dims or axis >= expanded_num_dims:
      raise ValueError("axis = %d not in [%d, %d)" %
                       (axis, -expanded_num_dims, expanded_num_dims))

  return gen_array_ops.pack(values, axis=axis, name=name)

@tf_export(v1=["losses.sparse_softmax_cross_entropy"])
def sparse_softmax_cross_entropy(
    labels, logits, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.
  `weights` acts as a coefficient for the loss. If a scalar is provided,
  then the loss is simply scaled by the given value. If `weights` is a
  tensor of shape `[batch_size]`, then the loss weights apply to each
  corresponding sample.
  Args:
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Unscaled log probabilities of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32` or
      `float64`.
    weights: Coefficients for the loss. This must be scalar or broadcastable to
      `labels` (i.e. same rank and each dimension is either 1 or the same).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `logits`. If `reduction` is
    `NONE`, this has the same shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shapes of `logits`, `labels`, and `weights` are
      incompatible, or if any of them are None.
  @compatibility(eager)
  The `loss_collection` argument is ignored when executing eagerly. Consider
  holding on to the return value or collecting losses via a `tf.keras.Model`.
  @end_compatibility
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "sparse_softmax_cross_entropy_loss",
                      (logits, labels, weights)) as scope:
    # As documented above in Args, labels contain class IDs and logits contains
    # 1 probability per class ID, so we expect rank(logits) - rank(labels) == 1;
    # therefore, expected_rank_diff=1.
    labels, logits, weights = _remove_squeezable_dimensions(
        labels, logits, weights, expected_rank_diff=1)
    losses = nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=logits,
                                                         name="xentropy")
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)
 
@tf_export(v1=["nn.sparse_softmax_cross_entropy_with_logits"])
def sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,  # pylint: disable=invalid-name
    labels=None,
    logits=None,
    name=None):
  """Computes sparse softmax cross entropy between `logits` and `labels`.
  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.
  **NOTE:**  For this operation, the probability of a given label is considered
  exclusive.  That is, soft classes are not allowed, and the `labels` vector
  must provide a single specific index for the true class for each row of
  `logits` (each minibatch entry).  For soft softmax classification with
  a probability distribution for each entry, see
  `softmax_cross_entropy_with_logits_v2`.
  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.
  A common use case is to have logits of shape
  `[batch_size, num_classes]` and have labels of shape
  `[batch_size]`, but higher dimensions are supported, in which
  case the `dim`-th dimension is assumed to be of size `num_classes`.
  `logits` must have the dtype of `float16`, `float32`, or `float64`, and
  `labels` must have the dtype of `int32` or `int64`.
  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**
  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
      `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
      must be an index in `[0, num_classes)`. Other values will raise an
      exception when this op is run on CPU, and return `NaN` for corresponding
      loss and gradient rows on GPU.
    logits: Per-label activations (typically a linear output) of shape
      `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float16`, `float32`, or
      `float64`. These activation energies are interpreted as unnormalized log
      probabilities.
    name: A name for the operation (optional).
  Returns:
    A `Tensor` of the same shape as `labels` and of the same type as `logits`
    with the softmax cross entropy loss.
  Raises:
    ValueError: If logits are scalars (need to have rank >= 1) or if the rank
      of the labels is not equal to the rank of the logits minus one.
  """
  _ensure_xent_args("sparse_softmax_cross_entropy_with_logits", _sentinel,
                    labels, logits)

  # TODO(pcmurray) Raise an error when the label is not an index in
  # [0, num_classes). Note: This could break users who call this with bad
  # labels, but disregard the bad results.

  # Reshape logits and labels to rank 2.
  with ops.name_scope(name, "SparseSoftmaxCrossEntropyWithLogits",
                      [labels, logits]):
    labels = ops.convert_to_tensor(labels)
    logits = ops.convert_to_tensor(logits)
    precise_logits = math_ops.cast(logits, dtypes.float32) if (dtypes.as_dtype(
        logits.dtype) == dtypes.float16) else logits

    # Store label shape for result later.
    labels_static_shape = labels.get_shape()
    labels_shape = array_ops.shape(labels)
    static_shapes_fully_defined = (
        labels_static_shape.is_fully_defined() and
        logits.get_shape()[:-1].is_fully_defined())
    if logits.get_shape().ndims is not None and logits.get_shape().ndims == 0:
      raise ValueError(
          "Logits cannot be scalars - received shape %s." % logits.get_shape())
    if logits.get_shape().ndims is not None and (
        labels_static_shape.ndims is not None and
        labels_static_shape.ndims != logits.get_shape().ndims - 1):
      raise ValueError("Rank mismatch: Rank of labels (received %s) should "
                       "equal rank of logits minus 1 (received %s)." %
                       (labels_static_shape.ndims, logits.get_shape().ndims))
    if (static_shapes_fully_defined and
        labels_static_shape != logits.get_shape()[:-1]):
      raise ValueError("Shape mismatch: The shape of labels (received %s) "
                       "should equal the shape of logits except for the last "
                       "dimension (received %s)." % (labels_static_shape,
                                                     logits.get_shape()))
    # Check if no reshapes are required.
    if logits.get_shape().ndims == 2:
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost

    # Perform a check of the dynamic shapes if the static shapes are not fully
    # defined.
    shape_checks = []
    if not static_shapes_fully_defined:
      shape_checks.append(
          check_ops.assert_equal(
              array_ops.shape(labels),
              array_ops.shape(logits)[:-1]))
    with ops.control_dependencies(shape_checks):
      # Reshape logits to 2 dim, labels to 1 dim.
      num_classes = array_ops.shape(logits)[array_ops.rank(logits) - 1]
      precise_logits = array_ops.reshape(precise_logits, [-1, num_classes])
      labels = array_ops.reshape(labels, [-1])
      # The second output tensor contains the gradients.  We use it in
      # _CrossEntropyGrad() in nn_grad but not here.
      cost, _ = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
          precise_logits, labels, name=name)
      cost = array_ops.reshape(cost, labels_shape)
      cost.set_shape(labels_static_shape)
      if logits.dtype == dtypes.float16:
        return math_ops.cast(cost, dtypes.float16)
      else:
        return cost
