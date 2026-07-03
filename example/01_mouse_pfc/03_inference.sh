source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/ 
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/12.1.0
module load libs/nccl/2.19.3_cuda12.0
module load libs/cudnn/9.8.0_cuda12
module load libs/openblas/0.3.26_gcc9.3.0

export DGLBACKEND=pytorch

conda activate stereotrack_2

python /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/example/01_mouse_pfc/03_inference_v1.py \
    --config /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/01_config_merfish_pfc_v3.yaml \
    --checkpoint /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/out/01_merfish_pfc_v3/checkpoints/last.ckpt 