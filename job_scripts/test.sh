#!/bin/bash -l
#SBATCH --job-name=test_egnn
#SBATCH --output=/home/emiled/Repos/e3_diffusion_for_molecules/outputs/test_out.log
#SBATCH --error=/home/emiled/Repos/e3_diffusion_for_molecules/outputs/test_err.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuRTX
#SBATCH --gpus=1

conda activate test_env
# jupyter notebook --no-browser --port=6161

# no coredumps
ulimit -S -c 0
ulimit -s unlimited

# set the scratch and results-directory
export TMPDIR=/tmp

echo $TMPDIR
echo $USER
echo $SLURM_JOBID
echo $TMPDIR/$USER/$SLURM_JOBID

# create working directory on node and copy input there
prologue()
{
  echo "training"
  mkdir -p $TMPDIR/$USER/$SLURM_JOBID
  cp -rf $SLURM_SUBMIT_DIR/* $TMPDIR/$USER/$SLURM_JOBID
  cd $TMPDIR/$USER/$SLURM_JOBID

  # create the directory in the current directory to store the results
  mkdir -p $SLURM_SUBMIT_DIR/result-$SLURM_JOBID
}

# run the program
runprogram()
{
  # run program
#   time python main_pi1m.py --exp $EXPERIMENT > log.txt 2>&1
  python main_pi1m.py --n_epochs 30 --exp_name edm_qm9_initialtest --n_stability_samples 1000 --wandb_usr emiledhifallah --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999
  sleep 86400
}

# do cleaning up after the job ended or has been cancelled
epilogue()
{
  MY_DATE=`date`
  echo "Copying files back at $MY_DATE from $SLURM_JOB_NODELIST"

  # copying files from scratch-directory back to home
  if cp -rf $TMPDIR/$USER/$SLURM_JOBID/* $SLURM_SUBMIT_DIR/result-$SLURM_JOBID
  then
    # deleting files from scratch-directories on the host
    echo "Deleting scratch data on the host"
    rm -rf $TMPDIR/$USER/$SLURM_JOBID
  else
    echo "IMPORTANT: Copying the data back from the host has failed, data is kept on the local nodes"
  fi
}

prologue
runprogram
epilogue

MY_DATE=`date`
echo "Job finished at $MY_DATE"

exit
