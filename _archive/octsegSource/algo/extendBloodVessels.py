def extendBloodVessels[bv, addWidth, multWidthTresh, multWidth]:
# EXTENDBLOODVESSELS Extends blood vessels by a constant factor or 
# multiplicative factor.
# 
# BVRES = ectendBloodVessels[BV, ADDWIDTH, MULTWIDTHTRESH, MULTWIDTH]
# BV: Blood vessel indices
# ADDWIDTH: Constant factor added left and right to the BV.
# MULTWIDTHTRESH: BV with size greater than this are extended
# multiplicatively.
# MULTWIDTH: Multiplicative extension factor. A factor of 0.5 extends the
# blood vessel position to start-0.5*bvWidth until +0.5*bvWidth, e.g.
# 0.5 results in a doubling of the length.
# BVRES: The extended BV indices.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: June 2010
# Revised comments: November 2015

lineWidth = size[bv, 2];

if nargin < 3:
    multWidth = 0;


# Extend the BV by a constant factor to the left or right
for k in 1:addWidth:
    for j in 1:size[bv,2]-1:
        if bv[j+1] != 0:
            bv[j] = 1;
        
    
    for j in size[bv,2]:-1:2:
        if bv[j-1] != 0:
            bv[j] = 1;
        
    


# Extend the BV by a multiplicative factor
if multWidth != 0:
    bvNew = zeros[1, lineWidth, 'uint8'];
    j = 1;
    while j <= lineWidth:
        if bv[j] == 1:
            a = j;
            while a <= lineWidth andand bv[a] == 1 :
                a = a + 1;
            
            a = a - 1;
            length = a - j + 1;
            if length > multWidthTresh:
                start = floor[j - multWidth * length];
                if start < 1:
                    start = 1;
                
                last = ceil[a + multWidth * length];
                if last > lineWidth:
                    last = lineWidth;
                
                vec = start:1:last;
                bvNew[vec] = 1;
                      
            j = a;
        
        j = j + 1;
    
    bv = bvNew;


bvRes = bv;