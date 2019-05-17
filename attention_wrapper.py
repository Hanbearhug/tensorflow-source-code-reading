
def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  """Convert to tensor and possibly mask `memory`.
  Args:
    memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
    memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
    check_inner_dims_defined: Python boolean.  If `True`, the `memory`
      argument's shape is checked to ensure all but the two outermost
      dimensions are fully defined.
  Returns:
    A (possibly masked), checked, new `memory`.
  Raises:
    ValueError: If `check_inner_dims_defined` is `True` and not
      `memory.shape[2:].is_fully_defined()`.
  """
  # 将memory转为tensor类型，如果memory_sequence_length非空，则也转化为tensor类型
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if memory_sequence_length is not None:
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
  # 如果检查是否定义了维度，则第2个维度(即除了batch和time维度以外的维度都不能未定义，即不能为None值)
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  # 如果没有定义memory_sequenth_length，则不需要进行mask操作。
  if memory_sequence_length is None:
    seq_len_mask = None
  # 否则进行mask操作，seq_len_mask是一个[batch_size,...]的向量，向量中为boo值，true为不需要进行mask，否则需要
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
    seq_len_batch_size = (
        memory_sequence_length.shape[0].value
        or array_ops.shape(memory_sequence_length)[0])
  # 这个函数应该是在做mask的操作，但是具体没看懂。。
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
    if memory_sequence_length is not None:
      message = ("memory_sequence_length and memory tensor batch sizes do not "
                 "match.")
      with ops.control_dependencies([
          check_ops.assert_equal(
              seq_len_batch_size, m_batch_size, message=message)]):
        seq_len_mask = array_ops.reshape(
            seq_len_mask,
            array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
        return m * seq_len_mask
    else:
      return m
return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)



def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  # 如果没有指定句子长度，直接返回得分
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  # 验证所有的句子长度是大于零的
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]):
  # score_mask为一个bool类型的张量，其中需要跳过的为False，否则为True
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[1])
  # 设置对于pad部分的score默认值，score_mask_value应该是一个标量
    score_mask_values = score_mask_value * array_ops.ones_like(score)
  # 如果不是pad部分则使用原有score中的值，否则使用设置的默认值填充
    return array_ops.where(score_mask, score, score_mask_values)
```

```
class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.
  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=None,
               name=None):
    """Construct base AttentionMechanism class.
    Args:
      query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
        must match the depth of `memory_layer`.  If `query_layer` is not
        provided, the shape of `query` must match that of `memory_layer`.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      probability_fn: A `callable`.  Converts the score and previous alignments
        to probabilities. Its signature should be:
        `probabilities = probability_fn(score, state)`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
        depth must match the depth of `query_layer`.
        If `memory_layer` is not provided, the shape of `memory` must match
        that of `query_layer`.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      name: Name to use when creating ops.
    """
    # 如果query_layer和memory_layer非空，则它们的类型应该为layer_base.Layer否则报错。
    if (query_layer is not None
        and not isinstance(query_layer, layers_base.Layer)):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    self.dtype = memory_layer.dtype
    # probability_fn是一个函数，用于将score转化为概率
    if not callable(probability_fn):
      raise TypeError("probability_fn must be callable, saw type: %s" %
                      type(probability_fn).__name__)
    # 如果mask的填充值为空，默认为-inf
    if score_mask_value is None:
      score_mask_value = dtypes.as_dtype(
          self._memory_layer.dtype).as_numpy_dtype(-np.inf)
    # 只使用句子里的值（去掉pad以后的值）进行概率计算
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(score, memory_sequence_length, score_mask_value),
            prev))
    # 如果命名域未设置则以BaseAttentionMechanismInit为默认命名域
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
    # 返回一个经过mask的tensor
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
    # 如果有memory层则经过memory层，否则直接返回
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
    # 得到batch_size以及句子长度
      self._batch_size = (
          self._keys.shape[0].value or array_ops.shape(self._keys)[0])
      self._alignments_size = (self._keys.shape[1].value or
                               array_ops.shape(self._keys)[1])

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  @property
  def state_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.
    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).
    The default behavior is to return a tensor of all zeros.
    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.
    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    # 使用全0张量初始化alignments
    max_time = self._alignments_size
    return _zero_state_tensors(max_time, batch_size, dtype)

  def initial_state(self, batch_size, dtype):
    """Creates the initial state values for the `AttentionWrapper` class.
    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).
    The default behavior is to return the same output as initial_alignments.
    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.
    Returns:
      A structure of all-zero tensors with shapes as described by `state_size`.
    """
    return self.initial_alignments(batch_size, dtype)


def _luong_score(query, keys, scale):
  """Implements Luong-style (multiplicative) scoring function.
  This attention has two forms.  The first is standard Luong attention,
  as described in:
  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025
  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.
  To enable the second form, call this function with `scale=True`.
  Args:
    query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    scale: Whether to apply a scale to the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  Raises:
    ValueError: If `key` and `query` depths do not match.
  """
  depth = query.get_shape()[-1]
  key_units = keys.get_shape()[-1]
  if depth != key_units:
    raise ValueError(
        "Incompatible or unknown inner dimensions between query and keys.  "
        "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
        "Perhaps you need to set num_units to the keys' dimension (%s)?"
        % (query, depth, keys, key_units, key_units))
  dtype = query.dtype

  # Reshape from [batch_size, depth] to [batch_size, 1, depth]
  # for matmul.
  query = array_ops.expand_dims(query, 1)

  # Inner product along the query units dimension.
  # matmul shapes: query is [batch_size, 1, depth] and
  #                keys is [batch_size, max_time, depth].
  # the inner product is asked to **transpose keys' inner shape** to get a
  # batched matmul on:
  #   [batch_size, 1, depth] . [batch_size, depth, max_time]
  # resulting in an output shape of:
  #   [batch_size, 1, max_time].
  # we then squeeze out the center singleton dimension.
  score = math_ops.matmul(query, keys, transpose_b=True)
  score = array_ops.squeeze(score, [1])
  # 参考bahdanau中使用v作为权重向量，这里也加入一个比例权重向量
  if scale:
    # Scalar used in weight scaling
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=init_ops.ones_initializer, shape=())
    score = g * score
return score

class LuongAttention(_BaseAttentionMechanism):
  """Implements Luong-style (multiplicative) attention scoring.
  This attention has two forms.  The first is standard Luong attention,
  as described in:
  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025
  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.
  To enable the second form, construct the object with parameter
  `scale=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="LuongAttention"):
    """Construct the AttentionMechanism mechanism.
    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional) The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the memory layer of the attention mechanism.
      name: Name to use when creating ops.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "luong_attention", [query]):
      score = _luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def _bahdanau_score(processed_query, keys, normalize):
  """Implements Bahdanau-style (additive) scoring function.
  This attention has two forms.  The first is Bhandanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, set `normalize=True`.
  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    normalize: Whether to normalize the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  # v是一个权重参数
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=init_ops.constant_initializer(math.sqrt((1. / num_units))),
        shape=())
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
  # 对v进行归一化可以加速收敛（思路来自于batch normalize）
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
  # soft alignment可以使得attention机制一起被纳入训练，（直接求内积得方式没办法在反向传播中被训练）
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [2])
  else:
return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])

class BahdanauAttention(_BaseAttentionMechanism):
  """Implements Bahdanau-style (additive) attention.
  This attention has two forms.  The first is Bahdanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="BahdanauAttention"):
    """Construct the Attention mechanism.
    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      # query_layer是一个dense层，query_depth一般是隐层神经元个数(hidden_size)
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def safe_cumprod(x, *args, **kwargs):
  """Computes cumprod of x in logspace using cumsum to avoid underflow.
  The cumprod function and its gradient can result in numerical instabilities
  when its argument has very small and/or zero values.  As long as the argument
  is all positive, we can instead compute the cumulative product as
  exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.
  Args:
    x: Tensor to take the cumulative product of.
    *args: Passed on to cumsum; these are identical to those in cumprod.
    **kwargs: Passed on to cumsum; these are identical to those in cumprod.
  Returns:
    Cumulative product of x.
  """
  # 直接连乘在数值上不够稳定，因此改为exp(log(x))的形式
  with ops.name_scope(None, "SafeCumprod", [x]):
    x = ops.convert_to_tensor(x, name="x")
    tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
    return math_ops.exp(math_ops.cumsum(
        math_ops.log(clip_ops.clip_by_value(x, tiny, 1)), *args, **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
  """Compute monotonic attention distribution from choosing probabilities.
  Monotonic attention implies that the input sequence is processed in an
  explicitly left-to-right manner when generating the output sequence.  In
  addition, once an input sequence element is attended to at a given output
  timestep, elements occurring before it cannot be attended to at subsequent
  output timesteps.  This function generates attention distributions according
  to these assumptions.  For more information, see `Online and Linear-Time
  Attention by Enforcing Monotonic Alignments`.
  Args:
    p_choose_i: Probability of choosing input sequence/memory element i.  Should
      be of shape (batch_size, input_sequence_length), and should all be in the
      range [0, 1].
    previous_attention: The attention distribution from the previous output
      timestep.  Should be of shape (batch_size, input_sequence_length).  For
      the first output timestep, preevious_attention[n] should be [1, 0, 0, ...,
      0] for all n in [0, ... batch_size - 1].
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'.
        * 'recursive' uses tf.scan to recursively compute the distribution.
          This is slowest but is exact, general, and does not suffer from
          numerical instabilities.
        * 'parallel' uses parallelized cumulative-sum and cumulative-product
          operations to compute a closed-form solution to the recurrence
          relation defining the attention distribution.  This makes it more
          efficient than 'recursive', but it requires numerical checks which
          make the distribution non-exact.  This can be a problem in particular
          when input_sequence_length is long and/or p_choose_i has entries very
          close to 0 or 1.
        * 'hard' requires that the probabilities in p_choose_i are all either 0
          or 1, and subsequently uses a more efficient and exact solution.
  Returns:
    A tensor of shape (batch_size, input_sequence_length) representing the
    attention distributions for each sequence in the batch.
  Raises:
    ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
  """
  # Force things to be tensors
  p_choose_i = ops.convert_to_tensor(p_choose_i, name="p_choose_i")
  previous_attention = ops.convert_to_tensor(
      previous_attention, name="previous_attention")
  if mode == "recursive":
    # Use .shape[0].value when it's not None, or fall back on symbolic shape
    batch_size = p_choose_i.shape[0].value or array_ops.shape(p_choose_i)[0]
    # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
    shifted_1mp_choose_i = array_ops.concat(
        [array_ops.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
    # Compute attention distribution recursively as
    # q[i] = (1 - p_choose_i[i - 1])*q[i - 1] + previous_attention[i]
    # attention[i] = p_choose_i[i]*q[i]
    attention = p_choose_i*array_ops.transpose(functional_ops.scan(
        # Need to use reshape to remind TF of the shape between loop iterations
        lambda x, yz: array_ops.reshape(yz[0]*x + yz[1], (batch_size,)),
        # Loop variables yz[0] and yz[1]
        [array_ops.transpose(shifted_1mp_choose_i),
         array_ops.transpose(previous_attention)],
        # Initial value of x is just zeros
        array_ops.zeros((batch_size,))))
  elif mode == "parallel":
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
    # Compute recurrence relation solution
    attention = p_choose_i*cumprod_1mp_choose_i*math_ops.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        clip_ops.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
  elif mode == "hard":
    # Remove any probabilities before the index chosen last time step
    p_choose_i *= math_ops.cumsum(previous_attention, axis=1)
    # Now, use exclusive cumprod to remove probabilities after the first
    # chosen index, like so:
    # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
    # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
    # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
    attention = p_choose_i*math_ops.cumprod(
        1 - p_choose_i, axis=1, exclusive=True)
  else:
    raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
  return attention


def _monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode,
                              seed=None):
  """Attention probability function for monotonic attention.
  Takes in unnormalized attention scores, adds pre-sigmoid noise to encourage
  the model to make discrete attention decisions, passes them through a sigmoid
  to obtain "choosing" probabilities, and then calls monotonic_attention to
  obtain the attention distribution.  For more information, see
  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784
  Args:
    score: Unnormalized attention scores, shape `[batch_size, alignments_size]`
    previous_alignments: Previous attention distribution, shape
      `[batch_size, alignments_size]`
    sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this larger
      than 0 will encourage the model to produce large attention scores,
      effectively making the choosing probabilities discrete and the resulting
      attention distribution one-hot.  It should be set to 0 at test-time, and
      when hard attention is not desired.
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'.  See the docstring for
      `tf.contrib.seq2seq.monotonic_attention` for more information.
    seed: (optional) Random seed for pre-sigmoid noise.
  Returns:
    A `[batch_size, alignments_size]`-shape tensor corresponding to the
    resulting attention distribution.
  """
  # Optionally add pre-sigmoid noise to the scores
  if sigmoid_noise > 0:
  # 不是很理解，猜测可能是加入独立噪声实际上增大了概率分布的方差，从而使得概率接近0，或者接近1的
  # 比例增加了，从而使得概率分布更像是one-hot，但是作用不明。
    noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                     seed=seed)
    score += sigmoid_noise*noise
  # Compute "choosing" probabilities from the attention scores
  if mode == "hard":
    # When mode is hard, use a hard sigmoid
    p_choose_i = math_ops.cast(score > 0, score.dtype)
  else:
    p_choose_i = math_ops.sigmoid(score)
  # Convert from choosing probabilities to attention distribution
return monotonic_attention(p_choose_i, previous_alignments, mode)

class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
  """Base attention mechanism for monotonic attention.
  Simply overrides the initial_alignments function to provide a dirac
  distribution, which is needed in order for the monotonic attention
  distributions to have the correct behavior.
  """

  def initial_alignments(self, batch_size, dtype):
    """Creates the initial alignment values for the monotonic attentions.
    Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length..., 0]
    for all entries in the batch.
    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.
    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    # 看上去像是初始化了一个单位矩阵，表示11对应关系
    max_time = self._alignments_size
    return array_ops.one_hot(
        array_ops.zeros((batch_size,), dtype=dtypes.int32), max_time,
dtype=dtype)
  
class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
  """Monotonic attention mechanism with Bahadanau-style energy function.
  This type of attention enforces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Since the attention scores are passed
  through a sigmoid, a learnable scalar bias parameter is applied after the
  score function and before the sigmoid.  Otherwise, it is equivalent to
  BahdanauAttention.  This approach is proposed in
  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               score_mask_value=None,
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               dtype=None,
               name="BahdanauMonotonicAttention"):
    """Construct the Attention mechanism.
    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
        for `_monotonic_probability_fn` for more information.
      sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
      score_bias_init: Initial value for score bias scalar.  It's recommended to
        initialize this to a negative value when the length of the memory is
        large.
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    # Set up the monotonic probability fn with supplied parameters
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(BahdanauMonotonicAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self._score_bias_init = score_bias_init

  def __call__(self, query, state):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(
        None, "bahdanau_monotonic_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=processed_query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, state)
    next_state = alignments
return alignments, next_state

class LuongMonotonicAttention(_BaseMonotonicAttentionMechanism):
  """Monotonic attention mechanism with Luong-style energy function.
  This type of attention enforces a monotonic constraint on the attention
  distributions; that is once the model attends to a given point in the memory
  it can't attend to any prior points at subsequence output timesteps.  It
  achieves this by using the _monotonic_probability_fn instead of softmax to
  construct its attention distributions.  Otherwise, it is equivalent to
  LuongAttention.  This approach is proposed in
  Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
  "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
  ICML 2017.  https://arxiv.org/abs/1704.00784
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               score_mask_value=None,
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               dtype=None,
               name="LuongMonotonicAttention"):
    """Construct the Attention mechanism.
    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
        for `_monotonic_probability_fn` for more information.
      sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
      score_bias_init: Initial value for score bias scalar.  It's recommended to
        initialize this to a negative value when the length of the memory is
        large.
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    # Set up the monotonic probability fn with supplied parameters
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(LuongMonotonicAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._score_bias_init = score_bias_init
    self._name = name

  def __call__(self, query, state):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).
    """
    with variable_scope.variable_scope(None, "luong_monotonic_attention",
                                       [query]):
      score = _luong_score(query, self._keys, self._scale)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, state)
    next_state = alignments
return alignments, next_state

class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "attention_state"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.
  Contains:
    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
    - `attention_state`: A single or tuple of nested objects
       containing attention mechanism state for each attention mechanism.
       The objects may contain Tensors or TensorArrays.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.
    The new state fields' shape must match original state fields' shape. This
    will be validated, and original fields' shape will be propagated to new
    fields.
    Example:
    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```
    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.
    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""
      if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
        return tensor_util.with_same_shape(old, new)
      return new

    return nest.map_structure(
        with_same_shape,
        self,
super(AttentionWrapperState, self)._replace(**kwargs)
