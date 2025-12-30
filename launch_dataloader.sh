python -m flame.data_hgt \
  --training.data_files /eagle/datascience/vsastry/projects/LatentTwinShared/hgt_all_new.npy \
  --training.input_len 8 \
  --training.target_len 2 \
  --training.stride 1 \
  --training.levels 7 \
  --training.batch_size 2 \
  --training.num_workers 0 
  #--training.pin_memory true \
  #--training.persistent_workers true
  #data_dir /eagle/datascience/vsastry/projects/LatentTwinShared/all_data/convert/nc_files \
