#!/bin/bash
#SBATCH --job-name=LLM_PDF
#SBATCH --output=/data/researchHome/pdou/hpc_cluster_report/reports/cluster_report_output.log
#SBATCH --error=/data/researchHome/pdou/hpc_cluster_report/reports/cluster_report_error.log
#SBATCH --nodelist=compute7

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2


#SBATCH --mem=0
#SBATCH --time=00:20:00

module load python/3.9.18
source /data/researchHome/pdou/hpc_cluster_report/my_env/bin/activate


export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29501

srun torchrun \
  --nproc-per-node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /data/researchHome/pdou/hpc_cluster_report/generate_report_new.py
