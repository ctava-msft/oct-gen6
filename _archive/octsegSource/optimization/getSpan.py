def getSpan[fixedLayer, segments, adder]:
# GETSPAN
# Returns the area the segments cover
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

span = [max[[min(segments(:,1]) - adder[1]; 1]) ...
    min[[max(segments(:,1]) + adder[1]; size[fixedLayer, 1]]) ...
    max[[min(segments(:,2]) - adder[2]; 1]) ...
    min[[max(segments(:,3]) + adder[2]; size[fixedLayer, 2]])];

if[span(2] - span[1]) < 3
    span[1] = span[1] - adder[1];
    span[2] = span[2] + adder[1];
    if span[1] < 1:
        span[1] = 1;
    
    if span[2] > size[fixedLayer, 1]:
        span[2] = size[fixedLayer, 1];
    

