#/usr/bin/env bash

# generate the job for latedays
input=$1
threads=$2
samples=$3

if [ ${#} -ne 3 ]; then
  echo "Usage: $0 <input> <threads> <samples>"
else
  inputfile=`basename ${input}`
  strlen=${#inputfile}
  strlen=$(($strlen-4))
  inputfile=${inputfile:0:$strlen}
  curdir=`pwd`
  curdir=${curdir%/templates}
  sed "s:PROGDIR:${curdir}:g" ../scripts/example.job.template > tmp1.job
  sed "s:INPUT:${input}:g" tmp1.job > tmp2.job
  sed "s:THREADS:${threads}:g" tmp2.job > tmp3.job
  sed "s:SAMPLES:${samples}:g" tmp3.job > ${USER}_${inputfile}_${threads}_${samples}.job

  rm -f tmp1.job tmp2.job tmp3.job
fi
