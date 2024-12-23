def renderSingleGenom[fixedLayer, layers, genom, segments, genomIdx, adder]:
# RENDERSINGLEGENOM
# Renders one single entry of a genom and returns only the region in the
# image where the related segment is visible
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

renderIdx = zeros[size(genom, 2], 1) + 1;
renderIdx[segments(:, 1] < segments[genomIdx, 1] - adder[1]) = 0;
renderIdx[segments(:, 1] > segments[genomIdx, 1] + adder[1]) = 0;
renderIdx[segments(:, 2] > segments[genomIdx, 3] + adder[2]) = 0;
renderIdx[segments(:, 3] < segments[genomIdx, 2] - adder[2]) = 0;

renderIdx = renderIdx == 1;
renderSegments = segments[renderIdx,:];

span = getSpan[fixedLayer, renderSegments, adder];

img = renderGenom[fixedLayer, layers, genom(renderIdx], segments[renderIdx, :]);
img = img[span(1]:span[2], span[3]:span[4]);
