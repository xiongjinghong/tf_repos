# steps
## step1 log to libsvm sample
sh get_join_sample.sh

## step2 stat sample & feature（可以跳过）
sh get_stat_feat.sh

## step3 remap feat_id（去掉低频特征，可以跳过）
sh get_remap_fid.sh

## step4 libsvm to tfrecords
python get_tfrecord.py --threads=10 --input_dir=./ --output_dir=./
