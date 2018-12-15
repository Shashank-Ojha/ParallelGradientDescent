#/usr/bin/env bash
# generate jobs in batch

threads=(1 2 4 8 16) # The number of threads
inputs=(OnlineNewsPopularity.txt) # The name of the input files
samples=(10) # The number of samples per thread

rm -f *.job

for f in ${inputs[@]}
do
    for t in ${threads[@]}
    do
      for s in ${samples[@]}
      do
	      ../scripts/generate_jobs.sh $f $t $s
      done
    done
done
