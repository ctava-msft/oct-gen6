def splitSegments[segments, mode, options, data]:
# SPLITSEGMENTS
# Splits the segments, if they are over a certain length:
# Two params:
# maxLength: Max. length of the resulting splitted segments
# thLength: Initial treshold for splitting a segment
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

segmentsSplitted = zeros[1,3];

if strcmp[mode, 'length']:
    maxLength = options[1];
    thLength = options[2];
    
k = 1;
for i in 1:size[segments, 1]:
    if segments[i, 3] - segments[i, 2] > thLength:
        idx = segments[i, 2]:maxLength:segments[i, 3];
        if idx[) != segments(i, 3]:
            idx[ + 1] = segments[i, 3];
        
        
        segmentsSplitted[k, 1] = segments[i, 1];
        segmentsSplitted[k, 2] = idx[1];
        segmentsSplitted[k, 3] = idx[2];
        k = k + 1;
        for s in 2:numel[idx]-1:
            segmentsSplitted[k, 1] = segments[i, 1];
            segmentsSplitted[k, 2] = idx[s] + 1;
            segmentsSplitted[k, 3] = idx[s+1];
            k = k + 1;
        
    else:
        segmentsSplitted[k, :] = segments[i, :];
        k = k + 1;
    


elif strcmp[mode, 'gradient']:
    thLength = options[1];
    thDiff = options[2];
    
    k = 1;
    for i in 1:size[segments, 1]:
        if segments[i, 3] - segments[i, 2] > thLength:
            dataSegment = zeros[size(data, 3], segments[i, 3] - segments[i, 2] + 1);
            for s in 1:size[data, 3]:
                dataSegment[s, :] = data[segments(i, 1], segments[i, 2]:segments[i, 3], s);
            
            
            dataSegmentsDiff = dataSegment[:,2:] - dataSegment[:, 1:-1];
            maxDiff = max[abs(dataSegmentsDiff], [], 1);
            idx = find[maxDiff > thDiff];
            idx = idx + segments[i,2];
            
            if numel[idx] == 0:
                segmentsSplitted[k, :] = segments[i, :];
                k = k + 1;
            else:
                if idx[1] != segments[i,2]:
                    idx = [segments[i,2] idx];
                
                
                if idx[) != segments(i, 3]:
                    idx[ + 1] = segments[i, 3];
                

                for s in 1:numel[idx]-2:
                    segmentsSplitted[k, 1] = segments[i, 1];
                    segmentsSplitted[k, 2] = idx[s];
                    segmentsSplitted[k, 3] = idx[s+1] - 1;
                    k = k + 1;
                
                segmentsSplitted[k, 1] = segments[i, 1];
                segmentsSplitted[k, 2] = idx[-1];
                segmentsSplitted[k, 3] = idx();
                k = k + 1;
            
        else:
            segmentsSplitted[k, :] = segments[i, :];
            k = k + 1;
        
    
