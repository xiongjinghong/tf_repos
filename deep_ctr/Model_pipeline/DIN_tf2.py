import numpy as np
import datetime
import itertools
import tensorflow as tf
from collections import namedtuple, OrderedDict
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

########################################################################
               #################数据预处理##############
########################################################################

# 定义输入数据参数类型
SparseFeat = namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed','embed_dim', 'dtype'])
DenseFeat = namedtuple('DenseFeat', ['name', 'pre_embed','reduce_type','dim', 'dtype'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'voc_size','hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim','maxlen', 'dtype'])

# 筛选实体标签categorical
DICT_CATEGORICAL = {"topic_id": [str(i) for i in range(0, 700)],
                    "keyword_id": [str(i) for i in range(0, 100)],                
           }


feature_columns = [SparseFeat(name="topic_id", voc_size=700, hash_size= None,share_embed=None, embed_dim=2, dtype='string'),
                   SparseFeat(name="keyword_id", voc_size=10, hash_size= None,share_embed=None, embed_dim=2, dtype='string'),
                    SparseFeat(name='client_type', voc_size=2, hash_size= None,share_embed=None, embed_dim=1,dtype='int32'),
                   SparseFeat(name='post_type', voc_size=2, hash_size= None,share_embed=None, embed_dim=1,dtype='int32'),
                    VarLenSparseFeat(name="follow_topic_id", voc_size=700, hash_size= None,share_embed='topic_id',weight_name = None, combiner= 'sum', embed_dim=2, maxlen=20,dtype='string'),
                    VarLenSparseFeat(name="all_topic_fav_7", voc_size=700, hash_size= None,share_embed='topic_id', weight_name = 'all_topic_fav_7_weight', combiner= 'sum', embed_dim=2, maxlen=5,dtype='string'),
                    VarLenSparseFeat(name="hist_topic_id", voc_size=700, hash_size= None, share_embed='topic_id', weight_name = None, combiner= None, embed_dim=2, maxlen=5,dtype='string'),
                   VarLenSparseFeat(name="hist_keyword_id", voc_size=10, hash_size= None, share_embed='keyword_id', weight_name = None, combiner= None, embed_dim=2, maxlen=5,dtype='string'),
#                    DenseFeat(name='client_embed',pre_embed='read_post_id', reduce_type='mean', dim=768, dtype='float32'),
                   ]

# 用户行为序列特征
history_feature_names = ['topic_id', 'keyword_id']

DEFAULT_VALUES = [[0],[''],[0.0],[0.0], [''], 
                  [''], [''],[''], [''],['']]
COL_NAME = ['act','client_id','client_type', 'post_type',"topic_id", 'follow_topic_id', "all_topic_fav_7",'hist_topic_id', 'keyword_id', 'hist_keyword_id']

def _parse_function(example_proto):
    
    item_feats = tf.io.decode_csv(example_proto, record_defaults=DEFAULT_VALUES, field_delim='\t')
    parsed = dict(zip(COL_NAME, item_feats))
    
    feature_dict = {}
    for feat_col in feature_columns:
        if isinstance(feat_col, VarLenSparseFeat):
            if feat_col.weight_name is not None:
                kvpairs = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                kvpairs = tf.strings.split(kvpairs, ':')
                kvpairs = kvpairs.to_tensor()
                feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                feat_vals = tf.reshape(feat_vals, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids= tf.strings.to_number(feat_ids, out_type=tf.int32) 
                feat_vals= tf.strings.to_number(feat_vals, out_type=tf.float32)
                feature_dict[feat_col.name] = feat_ids
                feature_dict[feat_col.weight_name] = feat_vals
            else:      
                feat_ids = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                feat_ids = tf.reshape(feat_ids, shape=[-1])
                if feat_col.dtype != 'string':
                    feat_ids= tf.strings.to_number(feat_ids, out_type=tf.int32) 
                feature_dict[feat_col.name] = feat_ids
    
        elif isinstance(feat_col, SparseFeat):
            feature_dict[feat_col.name] = parsed[feat_col.name]
            
        elif isinstance(feat_col, DenseFeat):
            if not feat_col.is_embed:
                feature_dict[feat_col.name] = parsed[feat_col.name]
            elif feat_col.reduce_type is not None: 
                keys = tf.strings.split(parsed[feat_col.is_embed], ',')
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(keys))
                emb = tf.reduce_mean(emb,axis=0) if feat_col.reduce_type == 'mean' else tf.reduce_sum(emb,axis=0)
                feature_dict[feat_col.name] = emb
            else:
                emb = tf.nn.embedding_lookup(params=ITEM_EMBEDDING, ids=ITEM_ID2IDX.lookup(parsed[feat_col.is_embed]))                
                feature_dict[feat_col.name] = emb
        else:
            raise "unknown feature_columns...."
        
        
    label = parsed['act']
    
    
    return feature_dict, label


pad_shapes = {}
pad_values = {}

for feat_col in feature_columns:
    if isinstance(feat_col, VarLenSparseFeat):
        max_tokens = feat_col.maxlen
        pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
        pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
        if feat_col.weight_name is not None:
            pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
            pad_values[feat_col.weight_name] = tf.constant(-1, dtype=tf.float32)

# no need to pad labels 
    elif isinstance(feat_col, SparseFeat):
        if feat_col.dtype == 'string':
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = '0'
        else:  
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0.0
    elif isinstance(feat_col, DenseFeat):
        if not feat_col.is_embed:
            pad_shapes[feat_col.name] = tf.TensorShape([])
            pad_values[feat_col.name] = 0.0
        else:
            pad_shapes[feat_col.name] = tf.TensorShape([feat_col.dim])
            pad_values[feat_col.name] = 0.0


pad_shapes = (pad_shapes, (tf.TensorShape([])))
pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))



filenames= tf.data.Dataset.list_files([
'./user_item_act_test_val.csv',  
])
dataset = filenames.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

batch_size = 2
dataset = dataset.map(_parse_function, num_parallel_calls=60)
dataset = dataset.repeat()
dataset = dataset.shuffle(buffer_size = batch_size) # 在缓冲区中随机打乱数据
dataset = dataset.padded_batch(batch_size = batch_size,
                               padded_shapes = pad_shapes,
                              padding_values = pad_values) # 每1024条数据为一个batch，生成一个新的Datasets
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 验证集
filenames_val= tf.data.Dataset.list_files(['./user_item_act_test_val.csv'])
dataset_val = filenames_val.flat_map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1))

val_batch_size = 2
dataset_val = dataset_val.map(_parse_function, num_parallel_calls=60)
dataset_val = dataset_val.padded_batch(batch_size = val_batch_size,
                               padded_shapes = pad_shapes,
                              padding_values = pad_values) # 每1024条数据为一个batch，生成一个新的Datasets
dataset_val = dataset_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

########################################################################
               #################自定义Layer##############
########################################################################


# 多值查找表稀疏SparseTensor >>  EncodeMultiEmbedding
class VocabLayer(Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (0) # mask成 0
            idx = tf.where(masks, idx, paddings)
        return idx
    
    def get_config(self):  
        config = super(VocabLayer, self).get_config()
        config.update({'mask_value': self.mask_value, })
        return config


class EmbeddingLookupSparse(Layer):
    def __init__(self, embedding, has_weight=False, combiner='sum',**kwargs):
        
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.has_weight = has_weight
        self.combiner = combiner
        self.embedding = embedding
    
    
    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)
        
    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding,sp_ids=idx, sp_weights=val, combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding,sp_ids=idx, sp_weights=None, combiner=self.combiner)
        return tf.expand_dims(combiner_embed, 1)
    
    def get_config(self):  
        config = super(EmbeddingLookupSparse, self).get_config()
        config.update({'has_weight': self.has_weight, 'combiner':self.combiner})
        return config


class EmbeddingLookup(Layer):
    def __init__(self, embedding, **kwargs):
        
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding
    
    
    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)
        
    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)
        return embed
    
    def get_config(self):  
        config = super(EmbeddingLookup, self).get_config()
        return config

    

# 稠密转稀疏 
class DenseToSparseTensor(Layer):
    def __init__(self, mask_value= -1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value
        

    def call(self, dense_tensor):    
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value , dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor
    
    def get_config(self):  
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config


# 自定义dnese层 含BN， dropout
class CustomDense(Layer):
    def __init__(self, units=32, activation='tanh', dropout_rate =0, use_bn=False, seed=1024, tag_name="dnn", **kwargs):
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.tag_name = tag_name
        
        super(CustomDense, self).__init__(**kwargs)

    #build方法一般定义Layer需要被训练的参数。    
    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='kernel_' + self.tag_name)
        self.bias = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='bias_' + self.tag_name)
        
        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()
            
        self.dropout_layers = tf.keras.layers.Dropout(self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(self.activation, name= self.activation + '_' + self.tag_name)
        
        super(CustomDense,self).build(input_shape) # 相当于设置self.built = True

    #call方法一般定义正向传播运算逻辑，__call__方法调用了它。    
    def call(self, inputs,training=None, **kwargs):
        fc = tf.matmul(inputs, self.weight) + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        out_fc = self.activation_layers(fc)
        
        return out_fc

    #如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法，保存模型不写这部分会报错
    def get_config(self):  
        config = super(CustomDense, self).get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_bn': self.use_bn, 
                       'dropout_rate': self.dropout_rate, 'seed': self.seed, 'name': self.tag_name})
        return config
    
    
class HashLayer(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(HashLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        zero = tf.as_string(tf.zeros([1], dtype='int32'))
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x
    def get_config(self, ):
        config = super(HashLayer, self).get_config()
        config.update({'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, })
        return config
    
#  Attention池化层   
class AttentionPoolingLayer(Layer):
    """
      Input shape
        - A list of three tensor: [query,keys,his_seq]
        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
        - his_seq is a 2D tensor with shape: ``(batch_size, T)``
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **att_activation**: Activation function to use in attention net.
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **hist_mask_value**: the mask value of his_seq.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, att_hidden_units=(80, 40), att_activation='sigmoid', weight_normalization=True,
                 mode="sum",hist_mask_value=0, **kwargs):

        self.att_hidden_units = att_hidden_units
        self.att_activation = att_activation
        self.weight_normalization = weight_normalization
        self.mode = mode
        self.hist_mask_value = hist_mask_value
        super(AttentionPoolingLayer, self).__init__(**kwargs)
        

    def build(self, input_shape):
        
        self.fc = tf.keras.Sequential()
        for unit in self.att_hidden_units:
            self.fc.add(layers.Dense(unit, activation=self.att_activation, name="fc_att_"+str(unit))) 
        self.fc.add(layers.Dense(1, activation=None, name="fc_att_out"))
        
        super(AttentionPoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        query, keys, his_seq = inputs
        # 计算掩码
        key_masks = tf.not_equal(his_seq, tf.constant(self.hist_mask_value , dtype=his_seq.dtype))
        key_masks = tf.expand_dims(key_masks, 1)
        
        # 1. 转换query维度，变成历史维度T 
        # query是[B, 1,  H]，转换到 queries 维度为(B, T, H)，为了让pos_item和用户行为序列中每个元素计算权重, tf.shape(keys)[1] 结果就是 T
        queries = tf.tile(query, [1, tf.shape(keys)[1], 1]) # [B, T, H]

        # 2. 这部分目的就是为了在MLP之前多做一些捕获行为item和候选item之间关系的操作：加减乘除等。
        # 得到 Local Activation Unit 的输入。即 候选queries 对应的 emb，用户历史行为序列 keys
        # 对应的 embed, 再加上它们之间的交叉特征, 进行 concat 后的结果
        din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
        # 3. attention操作，通过几层MLP获取权重，这个DNN 网络的输出节点为 1
        attention_score = self.fc(din_all) # [B, T, 1]
        # attention的输出, [B, 1, T]
        outputs = tf.transpose(attention_score, (0, 2, 1)) # [B, 1, T]

        # 4. 得到有真实意义的score
        if self.weight_normalization:
            # padding的mask后补一个很小的负数，这样后面计算 softmax 时, e^{x} 结果就约等于 0
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(outputs)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # 5. Activation，得到归一化后的权重
        if self.weight_normalization:
            outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        # 6. 得到了正确的权重 outputs 以及用户历史行为序列 keys, 再进行矩阵相乘得到用户的兴趣表征
        # Weighted sum，
        if self.mode == 'sum':
            # outputs 的大小为 [B, 1, T], 表示每条历史行为的权重,
            # keys 为历史行为序列, 大小为 [B, T, H];
            # 两者用矩阵乘法做, 得到的结果 outputs 就是 [B, 1, H]
            outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        else:
            # 从 [B, 1, H] 变化成 Batch * Time
            outputs = tf.reshape(outputs, [-1, tf.shape(keys)[1]]) 
            # 先把scores在最后增加一维，然后进行哈达码积，[B, T, H] x [B, T, 1] =  [B, T, H]
            outputs = keys * tf.expand_dims(outputs, -1) 
            outputs = tf.reshape(outputs, tf.shape(keys)) # Batch * Time * Hidden Size
        
        return outputs
   
    def get_config(self, ):

        config = {'att_hidden_units': self.att_hidden_units, 'att_activation': self.att_activation,
                  'weight_normalization': self.weight_normalization, 'mode': self.mode,}
        base_config = super(AttentionPoolingLayer, self).get_config()
        return config.update(base_config)
    

class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs,list):
            return inputs
        if len(inputs) == 1  :
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)


########################################################################
               #################定义输入帮助函数##############
#######################################################################
# 定义model输入特征
def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:    
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input([feat_col.dim], name=feat_col.name)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input([1], name=feat_col.name, dtype=feat_col.dtype)         
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input([None], name=feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input([None], name=feat_col.weight_name, dtype='float32')
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features

# 构造 自定义embedding层 matrix
def build_embedding_matrix(features_columns):
    embedding_matrix = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat) or isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
            vocab_size = feat_col.voc_size + 2
            embed_dim = feat_col.embed_dim
            if vocab_name not in embedding_matrix:
                embedding_matrix[vocab_name] = tf.Variable(initial_value=tf.random.truncated_normal(shape=(vocab_size, embed_dim),mean=0.0, 
                                                                           stddev=0.0, dtype=tf.float32), trainable=True, name=vocab_name+'_embed')
    return embedding_matrix

# 构造 自定义embedding层              
def build_embedding_dict(features_columns, embedding_matrix):
    embedding_dict = {}
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name     
            embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],name='emb_lookup_' + feat_col.name)
        elif isinstance(feat_col, VarLenSparseFeat):
            vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name 
            if feat_col.combiner is not None:
                if feat_col.weight_name is not None:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name],combiner=feat_col.combiner, has_weight=True, name='emb_lookup_sparse_' + feat_col.name)
                else:
                    embedding_dict[feat_col.name] = EmbeddingLookupSparse(embedding=embedding_matrix[vocab_name], combiner=feat_col.combiner, name='emb_lookup_sparse_' + feat_col.name) 
            else: 
                embedding_dict[feat_col.name] = EmbeddingLookup(embedding=embedding_matrix[vocab_name],name='emb_lookup_' + feat_col.name) 

    return embedding_dict


# dense 与 embedding特征输入
def input_from_feature_columns(features, features_columns, embedding_dict):
    sparse_embedding_list = []
    dense_value_list = []
    
    for feat_col in features_columns:
        if isinstance(feat_col, SparseFeat):
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys)(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=False)(_input)
                    
            embed = embedding_dict[feat_col.name](_input)
            sparse_embedding_list.append(embed)
        elif isinstance(feat_col, VarLenSparseFeat):      
            _input = features[feat_col.name]
            if feat_col.dtype == 'string':
                if feat_col.hash_size is None:
                    vocab_name = feat_col.share_embed if feat_col.share_embed else feat_col.name
                    keys = DICT_CATEGORICAL[vocab_name]
                    _input = VocabLayer(keys, mask_value='0')(_input)
                else:
                    _input = HashLayer(num_buckets=feat_col.hash_size, mask_zero=True)(_input)
            if feat_col.combiner is not None:
                input_sparse =  DenseToSparseTensor(mask_value=0)(_input)
                if feat_col.weight_name is not None:
                    weight_sparse = DenseToSparseTensor()(features[feat_col.weight_name])
                    embed = embedding_dict[feat_col.name]([input_sparse, weight_sparse])
                else:
                    embed = embedding_dict[feat_col.name](input_sparse)
            else:
                embed = embedding_dict[feat_col.name](_input)
                
            sparse_embedding_list.append(embed)
                
        elif isinstance(feat_col, DenseFeat):
            dense_value_list.append(features[feat_col.name])
            
        else:
            raise TypeError("Invalid feature column in input_from_feature_columns: {}".format(feat_col.name))
             
    return sparse_embedding_list, dense_value_list


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)
    
def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise "dnn_feature_columns can not be empty list"

        
def get_linear_logit(sparse_embedding_list, dense_value_list):
    
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = Add()([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        return dense_linear_layer
    else:
        raise "linear_feature_columns can not be empty list"

########################################################################
               #################定义模型##############
########################################################################

def DIN(feature_columns, history_feature_names, hist_mask_value,dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="sigmoid",
        att_weight_normalization=True, dnn_dropout=0, seed=1024):
    
    """Instantiates the Deep Interest Network architecture.
    Args:
        dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        history_feature_names: list,to indicate  sequence sparse field
        dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
        dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
        dnn_activation: Activation function to use in deep net
        att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
        att_activation: Activation function to use in attention net
        att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
        dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        seed: integer ,to use as random seed.
    return: A Keras model instance.
    """
    features = build_input_features(feature_columns)

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    query_feature_columns = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_feature_names:
            query_feature_columns.append(fc)
    key_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_names))
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            key_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    # 构建 embedding_dict
    embedding_matrix = build_embedding_matrix(feature_columns)
    embedding_dict = build_embedding_dict(feature_columns, embedding_matrix)

    query_emb_list, _ = input_from_feature_columns(features, query_feature_columns, embedding_dict)
    keys_emb_list, _ = input_from_feature_columns(features, key_feature_columns, embedding_dict)
    merge_dnn_columns = sparse_feature_columns + sparse_varlen_feature_columns + dense_feature_columns
    dnn_sparse_embedding_list, dnn_dense_value_list = input_from_feature_columns(features, merge_dnn_columns, embedding_dict)
    
    keys_emb = concat_func(keys_emb_list)
    query_emb = concat_func(query_emb_list)
    keys_seq = features[key_feature_columns[0].name]

    hist_attn_emb = AttentionPoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,hist_mask_value=hist_mask_value)([query_emb, keys_emb, keys_seq])
    dnn_input = combined_dnn_input(dnn_sparse_embedding_list+[hist_attn_emb], dnn_dense_value_list)

    # DNN
    for i in range(len(dnn_hidden_units)):
        if i == len(dnn_hidden_units) - 1:
            dnn_out = CustomDense(units=dnn_hidden_units[i],dropout_rate=dnn_dropout,
                                       use_bn=dnn_use_bn, activation=dnn_activation, name='dnn_'+str(i))(dnn_input)
            break
        dnn_input = CustomDense(units=dnn_hidden_units[i],dropout_rate=dnn_dropout,
                                     use_bn=dnn_use_bn, activation=dnn_activation, name='dnn_'+str(i))(dnn_input)  
    dnn_logit = Dense(1, use_bias=False, activation=None, kernel_initializer=tf.keras.initializers.glorot_normal(seed),name='dnn_logit')(dnn_out)
    output = tf.keras.layers.Activation("sigmoid", name="din_out")(dnn_logit)
    model = Model(inputs=inputs_list, outputs=output)

    return model



model = DIN(feature_columns, history_feature_names, hist_mask_value='0', dnn_use_bn=False,
        dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="sigmoid",
        att_weight_normalization=True, dnn_dropout=0, seed=1024)

model.compile(optimizer="adam", loss= "binary_crossentropy",  metrics=tf.keras.metrics.AUC(name='auc'))

log_dir = '/mywork/tensorboardshare/logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 write_graph=True,  # 是否存储网络结构图
                 write_images=True,# 是否可视化参数
                 update_freq='epoch',
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None,
                        profile_batch = 20)

total_train_sample =  10000
total_test_sample =    10
train_steps_per_epoch=np.floor(total_train_sample/batch_size).astype(np.int32)
test_steps_per_epoch = np.ceil(total_test_sample/val_batch_size).astype(np.int32)
history_loss = model.fit(dataset, epochs=3, 
          steps_per_epoch=train_steps_per_epoch,
          validation_data=dataset_val, validation_steps=test_steps_per_epoch, 
          verbose=1,callbacks=[tbCallBack])
