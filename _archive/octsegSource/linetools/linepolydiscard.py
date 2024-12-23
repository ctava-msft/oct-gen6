def linepolydiscard[in_line, goodpercent, polydegree, parts]:
# LINEPOLYDISCRAD Fits a polynome trough a line. Than spilts the line in
# aequidistant parts. A fixed percentage of each part is kept, the most
# distant points to the polynom are discarded.
# [SWLINE DELTA] = polyline[IN_LINE, GOODPERCENT, POLYDEGREE, PARTS]
# IN_LINE: The input line (Row vector according to OCTSEGs line definition)
# GOODPERCENT: How much (in %) of the points are kept?
# POLYDEGREE: Polynom degree used
# PARTS: In how many parts the line is split for comparison with the
#   polynom?
# SWLINE: Resulting line. 


# Set default parameters
if nargin < 4:
    parts = 5;

if nargin < 3:
    polydegree = 4;

if nargin < 2:
    goodpercent = 2/3;


# Fill gaps in the input line
entries = find[in_line];
noentries = in_line < 1;
line = interp1[entries, in_line(entries], 1:size[in_line,2], 'linear', 'extrap'); 

swline = zeros[1,size(in_line,2], 'single');
     
# Perform the polynom split
x = 1:size[line,2];
[p, S, mu] = polyfit[x, line, polydegree];
[ynew, not ] = polyval[p, x, S, mu];

diff = abs[ynew - line];

# Compute the part positions
segments = [1:floor[size(diff,2]/parts):size[diff,2]];
segments[ + 1] = size[diff,2];

# throw out the most distant points from the polynom.
for i in 1:size[segments, 2]-1:
    [not , IX] = sort[diff(segments(i]:segments[i+1]));
    goodVsIdx = sort[IX(1:floor( * goodpercent])) + segments[i] - 1;
    goodVs = line[goodVsIdx];
    swline[goodVsIdx] = goodVs;


swline[noentries] = 0;