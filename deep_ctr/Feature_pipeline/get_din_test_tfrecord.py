import tensorflow as tf
import numpy as np
import os
import random

# ===== 配置参数 =====
DATA_DIR = "./data"                # 数据根目录
NUM_TR_SAMPLES = 1000              # 训练样本数
NUM_TE_SAMPLES = 200               # 测试/验证样本数
FIELD_SIZE = 10                    # 必须与 FLAGS.field_size 一致
FEATURE_SIZE = 1000                # 必须与 FLAGS.feature_size 一致
MAX_SEQ_LEN = 8                    # 变长序列的最大长度（实际随机长度在 1~MAX_SEQ_LEN 之间）

# 创建目录
os.makedirs(os.path.join(DATA_DIR, "tr"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "te"), exist_ok=True)

def int64_feature(value):
    """包装 int 或 list 为 Int64List"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """包装 float 或 list 为 FloatList"""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def generate_sample():
    """生成一个样本的 feature dict"""
    # 标签 y (0/1)
    y = random.randint(0, 1)
    # 无用字段 z，随便填个 0.0
    z = 0.0

    # 定长 feat_ids (长度为 FIELD_SIZE)
    feat_ids = [random.randint(0, FEATURE_SIZE-1) for _ in range(FIELD_SIZE)]

    # 随机生成变长用户行为序列
    # u_cat
    seq_len_cat = random.randint(1, MAX_SEQ_LEN)
    u_catids = [random.randint(0, FEATURE_SIZE-1) for _ in range(seq_len_cat)]
    u_catvals = [random.random() for _ in range(seq_len_cat)]

    # u_shop
    seq_len_shop = random.randint(1, MAX_SEQ_LEN)
    u_shopids = [random.randint(0, FEATURE_SIZE-1) for _ in range(seq_len_shop)]
    u_shopvals = [random.random() for _ in range(seq_len_shop)]

    # u_int
    seq_len_int = random.randint(1, MAX_SEQ_LEN)
    u_intids = [random.randint(0, FEATURE_SIZE-1) for _ in range(seq_len_int)]
    u_intvals = [random.random() for _ in range(seq_len_int)]

    # u_brand
    seq_len_brand = random.randint(1, MAX_SEQ_LEN)
    u_brandids = [random.randint(0, FEATURE_SIZE-1) for _ in range(seq_len_brand)]
    u_brandvals = [random.random() for _ in range(seq_len_brand)]

    # 广告特征 (定长单值)
    a_catids = random.randint(0, FEATURE_SIZE-1)
    a_shopids = random.randint(0, FEATURE_SIZE-1)
    a_brandids = random.randint(0, FEATURE_SIZE-1)

    # 广告商品序列 (变长)
    seq_len_a_int = random.randint(1, MAX_SEQ_LEN)
    a_intids = [random.randint(0, FEATURE_SIZE-1) for _ in range(seq_len_a_int)]

    # 构造 Example
    feature = {
        "y": float_feature(y),
        "z": float_feature(z),
        "feat_ids": int64_feature(feat_ids),
        "u_catids": int64_feature(u_catids),
        "u_catvals": float_feature(u_catvals),
        "u_shopids": int64_feature(u_shopids),
        "u_shopvals": float_feature(u_shopvals),
        "u_intids": int64_feature(u_intids),
        "u_intvals": float_feature(u_intvals),
        "u_brandids": int64_feature(u_brandids),
        "u_brandvals": float_feature(u_brandvals),
        "a_catids": int64_feature(a_catids),
        "a_shopids": int64_feature(a_shopids),
        "a_brandids": int64_feature(a_brandids),
        "a_intids": int64_feature(a_intids),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecord(filename, num_samples):
    """写入 TFRecord 文件（使用 TF 1.4 兼容的 API）"""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for _ in range(num_samples):
            example = generate_sample()
            writer.write(example.SerializeToString())
    print(f"Wrote {num_samples} samples to {filename}")

if __name__ == "__main__":
    # 训练集：分成多个文件（可选）
    tr_file1 = os.path.join(DATA_DIR, "tr", "part-00000.tfrecord")
    write_tfrecord(tr_file1, NUM_TR_SAMPLES)

    # 测试/验证集
    te_file1 = os.path.join(DATA_DIR, "te", "part-00000.tfrecord")
    write_tfrecord(te_file1, NUM_TE_SAMPLES)
