# steps

1. log to libsvm sample  
   ```bash
   sh get_join_sample.sh
   ```
2. stat sample & feature（可以跳过）
 ```bash
   sh get_stat_feat.sh
   ```
3. remap feat_id（去掉低频特征，可以跳过）
 ```bash
   sh get_remap_fid.sh
   ```
4. libsvm to tfrecords
 ```bash
   python get_tfrecord.py --threads=10 --input_dir=./ --output_dir=./

   ```
