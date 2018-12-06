#/usr/bin/env bash

# generate the job for latedays
input=$1
threads=$2
alpha=$3


if [ ${#} -ne 2 ]; then
  echo "Usage: $0 <input> <threads>"
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
  sed "s:ALPHA:${alpha}:g" tmp3.job > ${USER}_${inputfile}_${threads}_${alpha}.job
  rm -f tmp1.job tmp2.job tmp3.job
fi
