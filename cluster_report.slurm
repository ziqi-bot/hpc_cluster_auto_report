#!/bin/bash
#SBATCH --job-name=ClusterReport
#SBATCH --output=/data/researchHome/pdou/hpc_cluster_report/reports/cluster_report_output.log
#SBATCH --error=/data/researchHome/pdou/hpc_cluster_report/reports/cluster_report_error.log
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1



source /data/researchHome/pdou/hpc_cluster_report/.venv/bin/activate
/data/researchHome/pdou/hpc_cluster_report/.venv/bin/python /data/researchHome/pdou/hpc_cluster_report/generate_report.py



#scontrol show jobid 
