"""
本文记录tensorflow的一些小工具的源码解读下面列出已阅读的源码
====================================================================
1.array_ops.slice：在指定起点（begin）截取指定大小的张量，size指在每一个维度上
分别截取的长度。
2. array_ops.stop_gradient：源码未找到，用于停止对于变量input的梯度计算。
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


