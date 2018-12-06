#/usr/bin/env bash
# generate jobs in batch

threads=(1 2 4 8 32) # The number of threads
inputs=(5000.txt) # The name of the input files
alpha=(0.01)



rm -f *.job

for f in ${inputs[@]}
do
    for t in ${threads[@]}
    do
      for a in ${alpha[@]}
      do
	      ../scripts/generate_jobs.sh $f $t $a
      done
    done
done
