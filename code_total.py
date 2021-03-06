"""
作者：韩方园
日期：2019-07-15
说明：FM、app2vec算法是我自己写的，DeepFM算法代码是github上的，地址：https://github.com/ChenglongChen/tensorflow-DeepFM，
关于LR和FM的速度，尽管他们都是线性时间复杂度（样本特征维度），但是sklearn的lr支持稀疏矩阵运算，而我搞了半天也不知道FM的稀疏矩阵运算怎么
做。。希望有大佬懂得教教我。app2vec实现了两种负采样方式，一种是原论文的词频的3/4幂采样，一种是log distribution采样，如果使用log distribution
采样，务必字典或者说word2index的顺序是index越小，词频越大，因为tensorflow采样源码在log采样那里是使用索引采样，且有索引越小词频越大的假设
。具体可以去看tensorflow的采样源码。与原论文不同的是我的app2vec在词袋模型里支持加权，加权张量为context_weight变量，需要注意的是我给
每一个词的权重设了下限，这是为了防止在某些极端情况下的loss前向传播中出现nan的情况目前尚未实现跳词模型和层次softmax，原因是跳词和词袋的效果差不多，
而层次softmax的时间复杂度是O（logN），但是负采样单次训练复杂度是O（1）（好吧我不想承认其实是因为我懒。。。写完这俩再写个load和save感觉要突破500
行了。。感觉自己写了个库出来。。以后有时间再说吧，另有任何不懂的或者改进的地方欢迎和本菜鸡讨论）

2019-07-16:更新AFM模型，加入attention机制（原论文中有源代码实现，我仅仅是练手，要用原汁原味的请去原论文查找。。），发现原论文居然用的是for循环
写出来的。。。估计比我的更慢，好处是应该不会那么吃内存。然而牺牲了效率使得模型复杂度为O(p^2)后果是模型巨慢，谁用谁知道。。。

2019-07-29：更新GCN半监督的图卷积神经网络，本质上是信息流在网络（社交网络等）基于拉普拉斯矩阵的传播过程，似乎也可以使用谱分析解释，然而这个待研究。。
支持非监督学习和半监督学习两种方式，半监督需要传入一个mask矩阵指明哪些是有标签节点。

注：后续会复现LINE、Node2Vec、GraRep等图嵌入模型，什么时候写看我什么时候想起来。。。。
"""

import pandas as pd
import numpy as np
import tensorflow as tf


class FM:
    def __init__(self, p=58, k=5):
        self.p = p
        self.k = k
        self._init_graph()

    def _init_graph(self):
        self.x = tf.placeholder(tf.float32, [None, None], name='x')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')
        self.weight = self._init_weight()
        self.linear_terms = tf.add(self.weight['w0'], tf.matmul(self.x, self.weight['w']), name='linear_terms')
        self.pair_interactions = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(
                    tf.matmul(self.x, tf.transpose(self.weight['v'])), 2),
                tf.matmul(tf.pow(self.x, 2), tf.transpose(tf.pow(self.weight['v'], 2)))
            ), axis=1, keepdims=True)
        self.y_hat = tf.nn.sigmoid(tf.add(self.linear_terms, self.pair_interactions), name='y_hat')

        self.lambda_w = tf.constant(0.001, name='lambda_w')
        self.lambda_v = tf.constant(0.001, name='lambda_v')

        self.l2_norm = tf.add(tf.reduce_sum(tf.multiply(self.lambda_w, tf.pow(self.weight['w'], 2))),
                         tf.reduce_sum(tf.multiply(self.lambda_v, tf.pow(self.weight['v'], 2))))

        self.log_loss = tf.losses.log_loss(self.y, self.y_hat, epsilon=1e-12)
        self.loss = tf.add(self.log_loss, self.l2_norm, name='loss')

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _init_weight(self):
        weight = {}
        weight['w0'] = tf.Variable(tf.random_normal([1], mean=0, stddev=0.01))
        weight['w'] = tf.Variable(tf.random_normal([self.p, 1], mean=0, stddev=0.01))
        weight['v'] = tf.Variable(tf.random_normal([self.k, self.p], mean=0, stddev=0.01))
        return weight

    def batcher(self, x, y=None, batch_size=64, is_train=True):
        step = int(len(x) / batch_size)
        if is_train:
            for i in range(step + 1):
                try:
                    x_batch = x[i * batch_size:(i + 1) * batch_size, :]
                    y_batch = y[i * batch_size:(i + 1) * batch_size]

                    yield x_batch, y_batch
                except:
                    x_batch = x[i * batch_size:, :]
                    y_batch = y[i * batch_size:]

                    yield x_batch, y_batch
        else:
            for i in range(step + 1):
                try:
                    x_batch = x[i * batch_size:(i + 1) * batch_size, :]
                    yield x_batch
                except:
                    x_batch = x[i * batch_size:, :]
                    yield x_batch

    def fit_on_batch(self, x_batch, y_batch):
        feed_dict = {
            self.x: x_batch,
            self.y: y_batch.reshape((-1,1))
        }
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def fit(self, x, y, epochs=50, batch_size=64):
        steps = int(x.shape[0]/batch_size) + 1
        for epoch in range(epochs):
            perm = np.random.permutation(x.shape[0])
            total_loss = []
            for x_batch, y_batch in tqdm_notebook(self.batcher(x[perm],y[perm], batch_size), total=steps, postfix=f'epoch: {epoch}', leave=False):
                total_loss.append(self.fit_on_batch(x_batch, y_batch))
            print(f'epoch:{epoch} loss: {np.mean(total_loss)}')

    def predict(self, x, batch_size=256):
        y = np.array([])
        for x_batch in self.batcher(x, is_train=False):
            y_pred = self.sess.run(self.y_hat, feed_dict={self.x:x_batch})
            if len(y)==0:
                y=y_pred
            else:
                y = np.vstack([y, y_pred])
        return y

def create_vocab_from_df(df, applist, min_freq=10):
    """
    从表格中创建app字典，并且过滤掉频率较低的app，将vocab按照词频由大到小排列
    :param df: 数据表格
    :param applist: 字符串，指代需要建立表格的applist
    :param min_freq: 最小频率
    :return:vocab(用于统计词频),index2id和id2index，其中剔除了频率低于指定阈值的app
    """
    word2index = {}
    word2index['pad'] = -1

    vocab = {}

    for app_stream in tqdm_notebook(df[applist]):
        for app in app_stream:
            if app in vocab.keys():
                vocab[app] += 1
            else:
                vocab[app] = 1
    select_app = [item[0] for item in vocab.items() if item[1] >= min_freq]
    drop_app = [item for item in vocab.items() if item[1] < min_freq]
    drop_count = sum([item[1] for item in drop_app])
    vocab['unknown'] = drop_count

    for app in [item[0] for item in drop_app]:
        vocab.pop(app)
    vocab = dict(sorted(vocab.items(), key=lambda r: r[1], reverse=True))
    count = 0
    for app in vocab.keys():
        if (app in word2index.keys()) or (count in word2index.values()):
            continue
        word2index[app] = count
        count += 1
    index2word = dict([(item[1], item[0]) for item in word2index.items()])
    return vocab, word2index, index2word


class app2vec:
    def __init__(self, sentences, vocab_size, embedding_size, word2index, index2word,
                 vocab, K=10, window_size=5, learning_rate=0.01, random_seed=79, distortion=0.75,
                 batch_size=64, is_weighted=False, sentences_gap_time=None, alpha=0.8, epislon=1e-3,
                 use_log_distribution=True):
        """
        aoo2vec的tensorflow实现
        :param sentences: 训练语料（即app序列）
        :param vocab_size: 字典大小
        :param embedding_size: 嵌入矩阵的维度
        :param word2index: 词到索引的转换字典
        :param index2word: 索引到词的转换字典
        :param vocab: 字典，如果use_log_distribution为真，则索引从小到大按照词频从大到小排列
        :param K: 负采样的采样数量
        :param window_size: 生成训练样本时的窗口大小（生成的训练样本长度为[batch_size, 2 * window_size]）
        :param learning_rate: 反向传播学习率
        :param random_seed: 随机种子
        :param distortion: 用于对均匀分布进行扭曲，word2vec原论文中使用distortion=0.75来做扭曲使得词频大的样本
        采样的频率不会过大，词频小的样本采样的频率不会过小
        :param batch_size:训练时的batch_size大小
        :param is_weighted:是否使用安装时间加权
        :param sentences_gap_time:与sentences规格一样，每一个app的安装时间
        :param alpha:
        :param epislon:如果加权，每一个词的最小权重，防止loss爆炸，出现nan的情况
        :param use_log_distribution:是否使用log分布抽样，如果是则vocab排序必须有序，如果否则默认
        使用原word2vec中调整的均匀分布进行采样。
        """
        if (is_weighted == True) and (sentences_gap_time is None):
            raise Exception('加权条件下必须给出对应的gap_time值！')

        self.random_seed = random_seed
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.window_size = window_size
        self.word2index = word2index
        self.index2word = index2word
        self.get_vector = None
        self.batch_size = batch_size
        self.unigrams = list(vocab.values())
        self.vocab = vocab
        self.distortion = 0.75
        self.K = K
        self.learning_rate = learning_rate
        self.epislon = epislon
        # self.word_vector 规格为[None, window_size, embedding_size]
        self.is_weighted = is_weighted
        self.sentences_gap_time = sentences_gap_time
        self.alpha = alpha
        self.use_log_distribution = use_log_distribution
        self.distortion = distortion

        self.sentences = self.preprocess(sentences)
        if self.is_weighted:
            self.word_vector, self.word_weight, self.label = self.transform_sentences()
        else:
            self.word_vector, self.label = self.transform_sentences()
        self.wordVectors = None
        print(f'训练集大小为{self.word_vector.shape[0]}')
        self._init_graph()

    def __len__(self):
        return len(self.vocab)

    def _get_vector_count(self):
        return self.word_vector.shape[0]

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.weights = self._initialize_weights()

            self.context = tf.placeholder(tf.int64, shape=[None, None], name='context')
            self.output = tf.placeholder(tf.int64, shape=[None, 1], name='output')

            self.context_v_vector = tf.nn.embedding_lookup(self.weights['embedding_v'], self.context,
                                                           name='context_v_vector')  # [None, window_size, embedding_size]
            self.output_u_vector = tf.nn.embedding_lookup(self.weights['embedding_u'], self.output,
                                                          name='output_u_vector')
            if self.is_weighted:
                self.context_weight = tf.placeholder(tf.float32, shape=[None, None, 1], name='context_weight')
                self.context_weight_norm = tf.div(self.context_weight,
                                                  tf.reduce_sum(self.context_weight, keepdims=True, axis=1),
                                                  name='context_weight_norm')

                self.weighted_context_v_vector = tf.multiply(self.context_v_vector, self.context_weight_norm,
                                                             name='weighted_context_v_vector')
                self.context_v_vector_average = tf.reduce_mean(self.weighted_context_v_vector, axis=1,
                                                               name='context_v_vector_average')

            else:
                self.context_v_vector_average = tf.reduce_mean(self.context_v_vector, axis=1,
                                                               name='context_v_vector_average')  # [None, embedding_size]
            # self.log_likelyhood_denominator = tf.reduce_sum(tf.exp(tf.matmul(self.context_v_vector_average, tf.transpose(self.u, [1, 0]))), name='log_likelyhood_denominator') # [None]
            # self.log_likelyhood_numerator = tf.diag_part(tf.matmul(self.output_u_vector, tf.transpose(self.context_v_vector_average, [1,0])), name='log_likelyhood_numerator') # [None]
            # self.log_likelyhood = tf.reduce_sum(tf.log(self.log_likelyhood_denominator) - tf.log(self.log_likelyhood_denominator), name='log_likelyhood')
            nce_biases = tf.Variable(tf.zeros([len(self)]))
            if not self.use_log_distribution:
                sample_values = candidate_sampling_ops.fixed_unigram_candidate_sampler(
                    true_classes=self.output,
                    num_true=1,
                    num_sampled=self.K,
                    unique=True,
                    range_max=len(self),
                    seed=self.random_seed,
                    distortion=self.distortion,
                    unigrams=self.unigrams
                )
            else:
                sample_values = None

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.weights['embedding_u'],
                    biases=nce_biases,
                    labels=self.output,
                    inputs=self.context_v_vector_average,
                    num_sampled=self.K,
                    num_classes=len(self),
                    sampled_values=sample_values
                )
            )

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self, word_vector, label, word_weight=None):
        steps = int(word_vector.shape[0] / self.batch_size) + 1
        for i in range(steps):
            try:
                word_batch = word_vector[i * self.batch_size:(i + 1) * self.batch_size, :]
                label_batch = label[i * self.batch_size:(i + 1) * self.batch_size, :]
                if self.is_weighted:
                    weight_batch = word_weight[i * self.batch_size:(i + 1) * self.batch_size, :]

            except:
                word_batch = word_vector[i * self.batch_size:, :]
                label_batch = label[i * self.batch_size:, :]
                if self.is_weighted:
                    weight_batch = word_weight[i * self.batch_size:, :]

            if self.is_weighted:
                yield word_batch, weight_batch, label_batch
            else:
                yield word_batch, label_batch

    def fit_on_batch(self, word_batch, label_batch, weight_batch=None):
        if self.is_weighted:
            feed_dict = {
                self.context: word_batch,
                self.output: label_batch,
                self.context_weight: weight_batch
            }
        else:
            feed_dict = {
                self.context: word_batch,
                self.output: label_batch
            }

        loss, opt = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def shuffle_word_label(self, word_vector, label, word_weight=None):
        index = np.random.permutation(range(word_vector.shape[0]))
        word_vector = word_vector[index, :]
        label = label[index]
        if self.is_weighted:
            word_weight = word_weight[index, :]
            return word_vector, word_weight, label
        else:
            return word_vector, label

    def fit(self, word_vector, label, word_weight=None, epochs=10):
        for epoch in range(epochs):
            steps = int(word_vector.shape[0] / self.batch_size) + 1
            total_loss = []
            if self.is_weighted:
                word_vector, word_weight, label = self.shuffle_word_label(word_vector, label, word_weight)

                for word_batch, weight_batch, label_batch in tqdm_notebook(
                        self.get_batch(word_vector, label, word_weight), total=steps, postfix=f'epoch: {epoch}',
                        leave=False):
                    if len(word_batch) == 0:
                        break
                    loss = self.fit_on_batch(word_batch, label_batch, weight_batch)
                    total_loss.append(loss)
            else:
                word_vector, label = self.shuffle_word_label(word_vector, label)

                for word_batch, label_batch in tqdm_notebook(self.get_batch(word_vector, label), total=steps,
                                                             postfix=f'epoch: {epoch}', leave=False):
                    loss = self.fit_on_batch(word_batch, label_batch)
                    total_loss.append(loss)
            print(f'epoch: {epoch} loss:{np.mean(total_loss)}')
        self.wordVectors = self.sess.run(self.weights['embedding_u'])
        self.word_norm = np.sqrt(np.square(self.wordVectors).sum(axis=1))
        return total_loss

    def preprocess(self, sentences):
        post_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            else:
                new_sentence = [word if (word in self.word2index.keys()) else 'unknown' for word in sentence]
                post_sentences.append(new_sentence)
        return post_sentences

    def transform_sentences(self):
        word_vector = []
        word_weight = []
        labels = []
        for sentence, gap_time in tqdm_notebook(zip(self.sentences, self.sentences_gap_time),
                                                total=len(self.sentences)):
            if isinstance(sentence, list):
                sentence = ['pad'] * self.window_size + sentence + ['pad'] * self.window_size
                if self.is_weighted:
                    gap_time = [0] * self.window_size + gap_time + [0] * self.window_size
                for word_index in range(self.window_size, len(sentence) - self.window_size):
                    window = [sentence[word_index + i] for i in range(-self.window_size, self.window_size + 1) if
                              i != 0]
                    vector = [self.word2index[word] for word in window]
                    if self.is_weighted:
                        vector_window = [gap_time[word_index + i] for i in
                                         range(-self.window_size, self.window_size + 1) if i != 0]
                        label_time = float(gap_time[word_index])
                        vector_weights = [self.epislon if vector_gap == 0 else max(self.epislon, self.alpha ** (
                                abs(float(vector_gap) - label_time) / 8640000)) for vector_gap in vector_window]

                    label = self.word2index[sentence[word_index]]
                    if len(vector_weights) == len(vector):
                        word_weight.append(vector_weights)
                        word_vector.append(vector)
                    else:
                        print('权重和向量长度不相等！')
                    labels.append(label)
        if self.is_weighted:
            return np.array(word_vector), np.expand_dims(np.array(word_weight), axis=2), np.array(labels).reshape(
                (-1, 1))
        else:
            return np.array(word_vector), np.array(labels).reshape((-1, 1))

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights['embedding_u'] = tf.Variable(
            tf.random_normal([self.vocab_size, self.embedding_size], 0.0, 0.01),
            name='embedding_u'
        )

        weights['embedding_v'] = tf.Variable(
            tf.random_normal([self.vocab_size, self.embedding_size], 0.0, 0.01),
            name='embedding_v'
        )
        return weights

    def get_weight(self, app):
        if isinstance(app, str):
            app = np.array([self.word2index[app]]).reshape((-1, 1))
        else:
            app = np.array([app]).reshape((-1, 1))
        app_weights = self.sess.run(self.output_u_vector, feed_dict={self.output: app})
        return app_weights

    def get_most_similar(self, positive=None, negative=None, k=10):
        if positive:
            positive_index = [self.word2index[app] for app in postive]
            positive_vector = self.wordVectors[positive_index, :].reshape((-1, self.embedding_size))
        else:
            positive_vector = np.zeros(self.wordVectors.shape[1]).reshape((-1, self.embedding_size))

        if negative:
            negative_index = [self.word2index[app] for app in negative]
            negative_vector = self.wordVectors[negative_index, :].reshape((-1, self.embedding_size))
        else:
            negative_vector = np.zeros(self.embedding_size).reshape((-1, self.embedding_size))

        app_vector = positive_vector.sum(axis=0) - negative_vector.sum(axis=0)
        app_norm = np.sqrt(np.square(app_vector).sum())

        cos_similarity = np.dot(self.wordVectors, app_vector) / (self.word_norm * app_norm)
        most_k = np.argsort(cos_similarity)[-k:]
        most_similar_word = [self.index2word[i] for i in most_k]
        return most_similar_word

class AFM:
    def __init__(self, field_size, feature_size, attention_hidden_size, embedding_size=50,
                 learning_rate=0.01):
        self.feature_size = feature_size # denote as M, size of the feature dictionary
        self.field_size = field_size # denote as F, size of the feature fields
        self.embedding_size = embedding_size # denote as K, size of the feature embedding
        self.attention_hidden_size = attention_hidden_size
        self.learning_rate = learning_rate
        self._init_graph()

    def _init_weight(self):
        weights = {}
        weights['embedding'] = tf.Variable(
            tf.truncated_normal([self.feature_size, self.embedding_size], 0, 0.01),
            name='embedding'
        )
        weights['feature_bias'] = tf.Variable(
            tf.truncated_normal([self.feature_size, 1], 0, 0.01),
            name='feature_bias'
        )
        weights['bias'] = tf.Variable(
            tf.truncated_normal([1], 0, 0.01),
            name='bias'
        )
        weights['attention_w'] = tf.Variable(
            tf.truncated_normal([self.embedding_size, self.attention_hidden_size], 0, 0.01),
            name='attention_w'
        )
        weights['attention_b'] = tf.Variable(
            tf.truncated_normal([1, self.attention_hidden_size], 0, 0.01),
            name='attention_b'
        )
        weights['attention_h'] = tf.Variable(
            tf.truncated_normal([self.attention_hidden_size, 1], 0, 0.01),
            name='attention_h'
        )
        weights['projection_attention'] = tf.Variable(
            tf.truncated_normal([self.embedding_size, 1], 0, 0.01),
            name='projection_attention'
        )
        return weights

    def _init_graph(self):
        tf.reset_default_graph()
        self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
        self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
        self.train_labels = tf.placeholder(tf.int32, [None, 1], name='train_labels')
        self.weights = self._init_weight()
        """
        =============================================================================================
        一次项计算
        """
        self.first_order_embedding = tf.nn.embedding_lookup(
            self.weights['feature_bias'],
            self.feature_index,
            name='first_order_embedding'
        ) # [batch_size, F, 1]
        self.first_order = tf.reshape(
            tf.reduce_sum(
                tf.multiply(
                    self.first_order_embedding,
                    tf.expand_dims(
                        self.feature_value,
                        -1
                    )
                ),
                axis=1
            ),
            shape=[-1,1],
            name='first_order'
        ) # [batch_size, 1]

        """
        =============================================================================================
        二次项计算，包括attention pool层的计算
        """

        self.feature_embedding = tf.nn.embedding_lookup(
            self.weights['embedding'],
            self.feature_index,
            name='feature_embedding'
        )
        self.feature_embedding_multi_value = tf.multiply(
            self.feature_embedding,
            tf.expand_dims(
                self.feature_value,
                axis=-1
            ),
            name='feature_embedding_multi_value'
        ) # [batch_size, feature_size, embedding_size]

        self.feature_embedding_multi_value_expand_dim = tf.expand_dims(
            tf.transpose(self.feature_embedding_multi_value, perm=[0,2,1]),
            axis=-1,
            name='feature_embedding_multi_value_expand_dim'
        ) # [batch_size, embedding_size, feature_size, 1]

        self.pair_wise_interaction = tf.subtract(
            tf.linalg.band_part(
                tf.matmul(
                    self.feature_embedding_multi_value_expand_dim,
                    tf.transpose(self.feature_embedding_multi_value_expand_dim, perm=[0, 1, 3, 2])
                ),
                num_lower=0,
                num_upper=-1,
            ),
            tf.linalg.band_part(
                tf.matmul(
                    self.feature_embedding_multi_value_expand_dim,
                    tf.transpose(self.feature_embedding_multi_value_expand_dim, perm=[0, 1, 3, 2])
                ),
                num_lower=0,
                num_upper=0,
            ),
            name='pair_wise_interaction'
        )


        upper_triangle_index = tf.where(
            tf.subtract(
                tf.linalg.band_part(
                    tf.ones_like(self.pair_wise_interaction),
                    num_lower=0,
                    num_upper=-1,
                ),
                tf.linalg.band_part(
                    tf.ones_like(self.pair_wise_interaction),
                    num_lower=0,
                    num_upper=0,
                ),
            )>0
        )

        self.pair_wise_interaction_flatten = tf.transpose(
            tf.reshape(
                tf.gather_nd(
                    self.pair_wise_interaction,
                    indices=upper_triangle_index,
                    name='pair_wise_interaction_flatten'
                ),
                shape=[-1, self.embedding_size, int(self.field_size * (self.field_size - 1)/2)]
            ),  # [batch_size, embedding_size, feature_size * (feature_size-1)]
            perm=[0,2,1]
        ) # [batch_size, feature_size * (feature_size-1), embedding_size]

        self.attention_net = tf.reshape(
            tf.matmul(
                tf.nn.relu(
                    tf.add(
                        tf.matmul(
                            tf.reshape(
                                self.pair_wise_interaction_flatten,
                                shape=(-1, self.embedding_size)
                            ),
                            self.weights['attention_w']
                        ),
                        self.weights['attention_b']
                    ),
                ),  # [batch_size, feature_size * (feature_size-1), attention_hidden_size]
                self.weights['attention_h']
            ),
            shape=(-1, int(self.field_size * (self.field_size - 1)/2), 1)
        ) # [batch_size, feature_size * (feature_size-1)/2, 1]

        self.attention_pool = tf.reduce_sum(
            tf.multiply(
                self.pair_wise_interaction_flatten,
                self.attention_net
            ),
            axis=1,
            name='attention_pool'
        ) # [batch_size, embedding_size]

        self.second_order = tf.matmul(
            self.attention_pool,
            self.weights['projection_attention']
        ) # [batch_size, 1]

        self.bias = self.weights['bias'] * tf.ones((tf.shape(self.feature_index)[0],1), dtype=tf.float32) # [batch_size, 1]
        self.y_hat = tf.nn.sigmoid(
            tf.add_n([self.bias, self.first_order, self.second_order]),
            name = 'y_hat'
        ) # [batch_size, 1]

        """
        ======================================================================================
        计算loss和反向传播
        """
        self.loss = tf.losses.log_loss(self.train_labels, self.y_hat, epsilon=1e-12)
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def batcher(self, feature_index, feature_value, y=None, batch_size=64, is_train=True):
        step = int(len(feature_index) / batch_size)
        if is_train:
            for i in range(step + 1):
                try:
                    feature_index_batch = feature_index[i * batch_size:(i + 1) * batch_size, :]
                    feature_value_batch = feature_value[i * batch_size:(i + 1) * batch_size, :]
                    y_batch = y[i * batch_size:(i + 1) * batch_size]

                    yield feature_index_batch, feature_value_batch, y_batch
                except:
                    feature_index_batch = feature_index[i * batch_size:, :]
                    feature_value_batch = feature_value[i * batch_size:, :]
                    y_batch = y[i * batch_size:]

                    yield feature_index_batch, feature_value_batch, y_batch
        else:
            for i in range(step + 1):
                try:
                    feature_index_batch = feature_index[i * batch_size:(i + 1) * batch_size, :]
                    feature_value_batch = feature_value[i * batch_size:(i + 1) * batch_size, :]

                    yield feature_index_batch, feature_value_batch
                except:
                    feature_index_batch = feature_index[i * batch_size:, :]
                    feature_value_batch = feature_value[i * batch_size:, :]

                    yield feature_index_batch, feature_value_batch

    def fit_on_batch(self, feature_index_batch, feature_value_batch, y_batch):
        feed_dict = {
            self.feature_index: feature_index_batch,
            self.feature_value: feature_value_batch,
            self.train_labels: y_batch.reshape((-1,1))
        }
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

    def fit(self, feature_index, feature_value, y, epochs=50, batch_size=64):
        steps = int(feature_index.shape[0] / batch_size) + 1
        for epoch in range(epochs):
            perm = np.random.permutation(feature_index.shape[0])
            total_loss = []
            for feature_index_batch, feature_value_batch, y_batch in tqdm_notebook(self.batcher(feature_index[perm], feature_value[perm], y[perm], batch_size), total=steps,
                                                  postfix=f'epoch: {epoch}', leave=False):
                total_loss.append(self.fit_on_batch(feature_index_batch, feature_value_batch, y_batch))
            print(f'epoch:{epoch} loss: {np.mean(total_loss)}')

    def predict(self, feature_index, feature_value, batch_size=256):
        y = np.array([])
        for feature_index_batch, feature_value_batch in self.batcher(feature_index, feature_value, is_train=False, batch_size=batch_size):
            y_pred = self.sess.run(self.y_hat, feed_dict={self.feature_index: feature_index_batch, self.feature_value: feature_value_batch})
            if len(y) == 0:
                y = y_pred
            else:
                y = np.vstack([y, y_pred])
        return y
class GCN:
    def __init__(self, D, A, y_true, mask_label, hidden_size=30, embedding_size=30, output_size=2, is_supervised=False):
        self.D = D
        self.A = A
        self.H = None
        self.y_true = y_true
        self.mask_label = mask_label
        self.is_supervised = is_supervised
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.preprocess()
        self._init_graph()

    def _init_graph(self):
        """
        初始化图，可定义图卷积层数
        :return:
        """
        self.y = tf.placeholder(shape=[None,1], name='y')
        self.mask = tf.placeholder(shape=[None, 1], name='mask')
        self.mask_norm = tf.div(self.mask, tf.reduce_mean(mask), name='mask_norm')
        self.A_bar_tensor = tf.constant(self.A_bar, name='A_bar_tensor')
        self.weight = self._init_weight()
        """
        ===========================================================================
        构建图卷积
        """
        if self.is_supervised:
            self.first_layer = tf.nn.relu(
                tf.matmul(
                    tf.matmul(self.A_bar_tensor, self.weight['embedding']),
                    self.weight['w0']
                ),
                name='first_layer'
            )
            self.second_layer = tf.matmul(
                tf.matmul(
                    self.A_bar_tensor,
                    self.first_layer
                ),
                self.weight['w1'],
                name='second_layer'
            )
            self.loss_pre_mask = tf.losses.sparse_softmax_cross_entropy(
                labels=self.y, logits=self.second_layer, name='loss_pre_mask'
            )
            self.loss_mask = tf.multiply(
                self.loss_pre_mask, self.mask_norm, name='loss_mask'
            )
            self.loss = tf.reduce_mean(self.loss_mask, name='loss')
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)

        else:
            self.first_layer = tf.nn.relu(
                tf.matmul(self.A_bar_tensor, self.weight['embedding']),
                name='first_layer'
            )
            self.second_layer = tf.nn.relu(
                tf.matmul(self.A_bar_tensor, self.first_layer),
                name='second_layer'
            )
            self.third_layer = tf.nn.relu(
                tf.matmul(self.A_bar_tensor, self.first_layer),
                name='third_layer'
            )
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _init_weight(self):
        weights = {}
        weights['embedding'] = tf.Variable(
            tf.truncated_normal(shape=(self.A.shape[0], self.embedding_size), mean=0, stddev=0.1),
            name='embedding',
        )
        weights['w0'] = tf.Variable(
            tf.truncated_normal(shape=(self.embedding_size, self.hidden_size),mean=0,stddev=0.1),
            name='w0'
        )
        weights['w1'] = tf.Variable(
            tf.truncated_normal(shape=(self.hidden_size, self.output_size), mean=0, stddev=0.1),
            name='w1'
        )
    def fit(self, epochs):
        for epoch in tqdm_notebook(range(epochs)):
            feed_dict = {
                self.y: self.y_true,
                self.mask: self.mask_label
            }
            loss, _ = self.sess.run(
                [self.loss, self.train_op],
                feed_dict=feed_dict
            )
            print(f'epoch{epoch}: loss:{loss}')
    def preprocess(self):
        self.A_tuta = self.A + np.identity(self.A.shape[0])
        self.D_tuta = np.diagflat(self.A_tuta.sum(axis=1))
        self.A_bar = np.dot(np.power(self.D_tuta, -0.5), self.A_tuta).dot(np.power(self.D_tuta, -0.5))
