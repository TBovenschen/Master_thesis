#bin/sh
# SGE: the job name
#$ -N diff_NT
#
# Choose node:
#$ -l hostname=science-bs35
# The requested run-time, expressed as (xxxx sec or hh:mm:ss)
#$ -l h_rt=00:50:00
#
# Memory limit
#$ -l h_vmem=25G
#
# SGE: your Email here, for job notification
#$ -M t.bovenschen@students.uu.nl
#
# SGE: when do you want to be notified (b : begin, e : end, s : error)?
#$ -m es
#
# SGE: ouput in the current working dir
#$ -wd /scratch/tycho
#
cd /nethome/4276361/thesis/code
python3 diff_NT.py
