#/usr/bin/env bash
#rm -f $(whoami)_*

curdir=`pwd`
for job in $curdir/../templates/*.job
do
    qsub $job
done
