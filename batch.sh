#!/bin/bash
#SBATCH --time=00:00:30
#SBATCH --nodes=1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --gres=gpu
#### IMPORTANT check which account and partition you can use
#### on the machine you are running on (you can use the 'myallocation' command)
module load cuda
cd /scratch/general/lustre/joseph_ditton/assignment4
#Run the program with our input
./hello