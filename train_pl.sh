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
export OMP_NUM_THREADS=4

conda activate stereotrack_2

cd /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
CONFIG_FILE="${1}"
echo "Using config file: ${CONFIG_FILE}"

. /home/share/huadjyin/home/zhoutao3/clashctl/scripts/cmd/clashctl.sh
clashon
export WANDB_API_KEY="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"
trap "clashoff" EXIT

# python example/03_Han_Neuron/preprocess_03.py --config "${CONFIG_FILE}"
python train_pl.py --config "${CONFIG_FILE}"

# nohup bash train_pl.sh /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v4.yaml > /home/share/huadjyin/home/zhoutao3/tracks/stereoTrack/config/02_config_Chen_Cell_v4.train.log 2>&1 &