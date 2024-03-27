#!/bin/bash

#SBATCH --output=./out/output_%a.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --array=0-999

module load python
source activate /project/vitelli/ml_venv
cd /home/schmittms/exact_IB/4Well/

idx=$SLURM_ARRAY_TASK_ID

n=0
#for x in {100,}" "{2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}" "{0.1,0.15,0.2,0.25,0.3,0.35,0.4}" "{1.3,}" "{2400,}" "{1.,}" "{10,}" "{log,}" "{1000,}" "{0.01,}" "{0.0001,}" "{2,3,4,5,6}" "{2,8}" "{500,}" "{steady,}; 
#for x in {100,}" "{3.0,}" "{0.1,0.15,0.2,0.25,0.3,0.35,0.4}" "{1.3,}" "{2400,}" "{1.,}" "{50,}" "{log,}" "{1000,}" "{0.01,}" "{0.0001,}" "{4,}" "{2,8}" "{500,}" "{steady,}; 
for x in {100,}" "{2.5,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0}" "{0.1,0.15,0.2,0.25,0.3,0.35,0.4}" "{1.3,}" "{2400,}" "{1.,}" "{50,}" "{log,}" "{1000,}" "{0.01,}" "{0.0001,}" "{2,3,4,5,6}" "{16,32}" "{500,}" "{steady,}; 
do
    ar=($x)
    N=${#ar[@]}
    if [[ "$N" == "15" ]]; then
            if [[ "$n" == "$idx" ]]; then
                n_steps=${ar[0]}
                mu=${ar[1]}
                sigma=${ar[2]}
                x_max=${ar[3]}
                beta_max=${ar[4]}
                beta_min=${ar[5]}
                beta_steps=${ar[6]}
                beta_loglin=${ar[7]}
                iter_steps=${ar[8]}
                randomness_pre=${ar[9]}
                randomness=${ar[10]}
                C_H=${ar[11]}
                L_B=${ar[12]}
                nbins=${ar[13]}
                steady=${ar[14]}
            fi
            n=$((n+1))
    fi
done


echo $n_steps
echo $mu
echo $sigma
echo $x_max
echo $beta_max
echo $beta_min
echo $beta_steps
echo $beta_loglin
echo $iter_steps
echo $randomness_pre
echo $randomness
echo $C_H
echo $L_B


python perform_iterative_IB.py \
        --n_steps $n_steps \
	--mu $mu \
        --sigma $sigma \
        --x_max $x_max \
        --beta_max $beta_max \
        --beta_min $beta_min \
        --beta_steps $beta_steps \
        --beta_loglin $beta_loglin \
        --iter_steps $iter_steps \
        --randomness_pre $randomness_pre \
        --randomness $randomness \
        --C_H $C_H \
        --L_B $L_B \
        --nbins $nbins \
        --steady $steady \
