source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/ 
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate /home/HPCBase/PACKAGE/linlei/miniforge3/envs/rapids_singlecell_26.02
export LD_PRELOAD=/home/HPCBase/PACKAGE/linlei/miniforge3/envs/rapids_singlecell_26.02/lib/libgomp.so.1

python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/04_spatial_inte.py \
  --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v3.yaml  \
  --cell_resolution 3.5 \
  --niche_resolution 1 \
  --n_neighbors 10 \
  --device cuda:0