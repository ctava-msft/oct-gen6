def mutateGenom[genom, numMut, values]:
# MUTATEGENOM
# Mutates a genom
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

genomIdx = randperm[numel(genom]);
genomIdx = genomIdx[1:numMut];

genomMut = genom;

for g in 1:numel[genomIdx]:
    pos = randi[size(values, 2] - 1);
    valuesFliped = values[values != genomMut(genomIdx(g]));
    genomMut[genomIdx(g]) = valuesFliped[pos];

