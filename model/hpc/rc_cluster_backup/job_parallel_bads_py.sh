#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=myTest
#SBATCH --mail-type=END
#SBATCH --mail-user=db4058@nyu.edu
#SBATCH --output=slurm_%j.out


source /home/db4058/road_construction/env_test/bin/activate

cd /home/db4058/road_construction/model/

module load matlab/2020a

python bads_start.py