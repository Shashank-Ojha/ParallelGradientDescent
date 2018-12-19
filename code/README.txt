Xeon Phi Machines:

    Logging in:
        ssh latedays.andrew.cmu.edu
    Directory:
        /home/ANDREWID/


NVIDIA GeForce GTX 1080 GPU:

    Logging in:
        ssh ghc27.ghc.andrew.cmu.edu
    Directory:
        /afs/andrew.cmu.edu/usr7/ANDREWID/private”


---------------------------------------------------------------------------

Running openMP files:

  make clean
  make
  make submit

  Edit parameters by modifying openMP/scripts/batch_generate.sh

Running cuda files:

  make clean
  make
  ./cudaGD -f Filename -b NumBlocks -t threadsPerBlock -s samplesPerThread


Each directory has a symlink to the outer data directory

---------------------------------------------------------------------------

test_model.py is a python script to verify the MSE given a dataset and estimator
