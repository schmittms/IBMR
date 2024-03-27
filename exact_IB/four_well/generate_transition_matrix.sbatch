#!/bin/bash

#SBATCH --output=./out/output_%a.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --array=0-83




module load python
source activate /project/vitelli/ml_venv
cd /home/schmittms/exact_IB/4Well/

#SBATCH --qos=vitelli
idx=$(($SLURM_ARRAY_TASK_ID-2))
idx=$SLURM_ARRAY_TASK_ID


#echo $idx
n=0
#for x in {100000,}" "{100,}" "{2000,}" "{12,11,10,9,8,7,6,5,4.5,4,3.5,3,2.5,2}" "{0.1,0.15,0.2,0.25,0.3,0.4}" "{1.3,}" "{100,}; 
#for x in {100000,}" "{100,}" "{2000,}" "{3,2.5}" "{0.125,0.175,0.225,0.275,0.325,0.35,0.375}" "{1.3,}" "{100,}; 
#for x in {100000,}" "{100,}" "{2000,}" "{3,2.5}" "{0.3,}" "{1.3,}" "{100,}; 
for x in {200000,}" "{100,}" "{4000,}" "{3,2.5}" "{0.1,0.125,0.15,0.175,0.2,0.225,0.25}" "{1.3,}" "{100,}; 
do
    ar=($x)
    N=${#ar[@]}
    if [[ "$N" == "7" ]]; then
    	    echo $N $n 
            if [[ "$n" == "$idx" ]]; then
                n_init_cond=${ar[0]}
                n_steps=${ar[1]}
                n_runs=${ar[2]}
                mu=${ar[3]}
                sigma=${ar[4]}
                x_max=${ar[5]}
                n_bins=${ar[6]}

            fi
            n=$((n+1))
    fi
done


echo $n_init_cond $n_steps $n_runs $mu $sigma $x_max $n_bins


python generate_transition_matrix.py \
	--n_init_cond $n_init_cond \
	--n_steps $n_steps \
	--n_runs $n_runs \
	--mu $mu \
	--sigma $sigma \
	--x_max $x_max \
	--n_bins $n_bins 
