CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --mode val \
  --model dsp \
  --loss dsp \
  --features_dir data_argo/features/ \
  --obs_len 20 \
  --pred_len 30 \
  --shuffle \
  --use_cuda True \
  --model_path saved_models/ckpt_dsp_220804.tar \
  --adv_cfg_path config.dsp_cfg