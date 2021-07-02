#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem=8GB
#SBATCH --array=1-10
#SBATCH --job-name=arrayBADS
#SBATCH --mail-type=END
#SBATCH --mail-user=db4058@nyu.edu
#SBATCH --output=array_bads_%A_%a.out
#SBATCH --error=array_bads_%A_%a.err


source /home/db4058/road_construction/rc_env/bin/activate

module load matlab/2020b

cd /home/db4058/road_construction/model/

cat<<EOF | matlab -nodisplay
bads_run(1, $SLURM_JOB_ID, $SLURM_ARRAY_TASK_ID, 1)
EOF