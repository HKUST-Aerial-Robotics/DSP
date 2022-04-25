echo "-- Processing val set..."
python argo_data/run_preprocess.py --mode val \
  --data_dir ~/data/dataset/argo_motion_forecasting/val/data/ \
  --save_dir argo_data/features/ \
  --small

# echo "-- Processing train set..."
# python argo_data/run_preprocess.py --mode train \
#   --data_dir ~/data/dataset/argo_motion_forecasting/train/data/ \
#   --save_dir argo_data/features/ \
#   --small

# echo "-- Processing test set..."
# python argo_data/run_preprocess.py --mode test \
#   --data_dir ~/data/dataset/argo_motion_forecasting/test_obs/data/ \
#   --save_dir argo_data/features/ \
#   --small