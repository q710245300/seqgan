import tensorflow as tf

encoder_outputs = tf.constant([[[1, 3, 1], [2, 3, 2]], [[2, 3, 4], [2, 3, 2]]])
print(encoder_outputs.get_shape())  # (2, 2, 3)
# 将batch内的每个样本复制3次, tile_batch() 的第2个参数是一个 int 类型数据
z4 = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=3)

print(z4.get_shape())