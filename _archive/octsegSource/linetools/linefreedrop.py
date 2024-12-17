def linefreedrop[lines, nh, minNearPoints, savelength]:
# LINEFREEDROP Removes lines, that have no neighbouring lines left or
# right. A neighborhood around the endpoints of each line is checked, and
# if enough linepoints lie within this region on both sides, it is kept.:
# Lines over a certain lenght are kept in every case.
# LONGLINES = linefreedrop[LINES, NH, MINNEARPOINTS, SAVELENGTH]
# LINES: Linematrix according to OCTSEGs standard with one additional
#   assumption: The line segment has no breaks, that means 0 can appear 
#   only at the left and right side of the line, not in between.
# NH: Neighbourhoodwidth to the left and right, top and down
# MINNEARPOINTS: minimum number of points in the left or right 
#   neighbourhood of the line for the line to be not discarded
# SAVELENGTH: Lines above that length are saved even if they don't have:
#   any neighbours
# LONGLINES: The resulting lines with neighbors.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg
#
# First final Version: June 2010

# Default parameters
if nargin < 4:
    savelength = 20;

if nargin < 3:
    minNearPoints = 1;

if nargin < 2:
    nh = 5;


longlines = [];

# The line is extended to the left and right to be on the save side
# regarding the neighborhood
symml = 10;
linesS = [lines[:,symml+1:-1:2] lines lines[:,-1:-1:-symml]];

linesindex = lines > 0;
linesum = sum[linesindex'];

# Outer loop: Go trough all lines
for i in 1:size[lines,1]:
    
    # Is there are line?
    entries = find[lines(i,:]);

    if size[entries] > 0      :
        
        # Check left neighbors
        j = entries[1];
        nbCount = 0;
        for x in 1:size[lines,1]:
            for y in j-nh+symml:j+symml:
                if (linesS[x,y] >= lines[i,j]-nh) andand (linesS[x,y] <= lines[i,j]+nh):
                    nbCount = nbCount + 1;
                
            
        
        leftnb = nbCount;
        
        # Check right neigbors
        j = entries();
        for x in 1:size[lines,1]:
            for y in j+symml:j+nh+symml:
                if (linesS[x,y] >= lines[i,j]-nh) andand (linesS[x,y] <= lines[i,j]+nh):
                    nbCount = nbCount + 1;
                
            
        

        rightnb = nbCount;

        # Check condition
        if (leftnb > minNearPoints andand rightnb > minNearPoints) oror linesum[i] >= savelength:
            longlines = [longlines; lines[i,:]];
        
        
    


