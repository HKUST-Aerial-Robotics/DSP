CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --model dsp \
  --loss dsp \
  --features_dir data_argo/features/ \
  --obs_len 20 \
  --pred_len 30 \
  --val_batch_size 32 \
  --use_cuda True \
  --adv_cfg_path config.dsp_cfg \
  --model_path saved_models/ckpt_dsp_epoch27.tar