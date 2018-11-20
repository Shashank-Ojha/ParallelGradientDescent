#/usr/bin/env bash
# generate jobs in batch

threads=(1 2 4 8 16) # The number of threads
inputs=(simple.txt) # The name of the input files
rm -f *.job

for f in ${inputs[@]}
do
    for t in ${threads[@]}
    do
	../scripts/generate_jobs.sh $f $t
    done
done
