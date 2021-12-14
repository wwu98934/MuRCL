#!/bin/sh

echo "pre-training via train_MuRCL.py"
for STAGE in 1 2; do
  python ../train_MuRCL.py \
    --dataset Camelyon16 \
    --data_csv path/to/data_csv.csv \
    --feat_size 1024 \
    --preload \
    --train_stage ${STAGE} \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --batch_size 128 \
    --epochs 100 \
    --backbone_lr 0.0001 \
    --fc_lr 0.00005 \
    --patience 10 \
    --arch CLAM_SB \
    --device 3 \
    --exist_ok
done
python ../train_MuRCL.py \
  --dataset Camelyon16 \
  --data_csv path/to/data_csv.csv \
  --feat_size 1024 \
  --preload \
  --train_stage 3 \
  --T 6 \
  --scheduler CosineAnnealingLR \
  --batch_size 128 \
  --epochs 100 \
  --backbone_lr 0.00005 \
  --fc_lr 0.00001 \
  --patience 10 \
  --arch CLAM_SB \
  --device 3 \
  --exist_ok
