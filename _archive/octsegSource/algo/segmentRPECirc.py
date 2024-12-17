def segmentRPE[bscan, params, medline]:
# SEGMENTRPE Segments the RPE from a BScan. Intended for the use on
# circular OCT-Bscans
# 
# RPEAUTO = segmentRPE[BSCAN, PARAMS, MEDLINE]
# 
# RPEAUTO: Automated segmentation of the RPE
# BSCAN: Unnormed BScan image 
# PARAMS:   Parameter struct for the automated segmentation
#   In this function, the following parameters are currently used:
#   RPE_SEGMENT_MEDFILT1 First median filter values (in z and x direction).
#   Preprocessing before finding extrema. (suggestion: 5 7)
#   RPE_SEGMENT_MEDFILT2 Second median filter values (in z and x direction).
#   Preprocessing before finding extrema. Directly applied after the first
#   median filter. (suggestion: Use the same settings)
#   RPE_SEGMENT_LINESWEETER1 Linesweeter smoothing values before blood
#   vessel region removal
#   RPE_SEGMENT_LINESWEETER2 Linesweeter smoothing values after blood
#   vessel region removal. This is the final smoothing applied to the RPE.
#   RPE_SEGMENT_POLYDIST Remove points with this (pixel) distance from
#   the RPE after the polynomial fit.
#   RPE_SEGMENT_POLYNUMBER Polynomial degree to fit after extrema
#   finding.
# 
# The algorithm (of which this function is a part) is described in 
# Markus A. Mayer, Joachim Hornegger, Christian Y. Mardin, Ralf P. Tornow:
# Retinal Nerve Fiber Layer Segmentation on FD-OCT Scans of Normal Subjects
# and Glaucoma Patients, Biomedical Optics Express, Vol. 1, Iss. 5, 
# 1358-1383 (2010). Note that modifications have been made to the
# algorithm since the paper publication.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: June 2010
# Revised comments: November 2015

# 1) Normalize the intensity values
bscan[bscan > 1] = 0; 
snBScan = splitnormalize[bscan, params, 'ipsimple opsimple soft', medline];
snBScan = removebias[snBScan, params];

# 2) A simple noise removal
snBScan = medfilt2[snBScan, params.RPE_SEGMENT_MEDFILT1];
snBScan = medfilt2[snBScan, params.RPE_SEGMENT_MEDFILT2];

# 3) Edge detection along A-Scans (taking the sign of the derivative into
# account)
rpe = findRetinaExtrema(snBScan, params, 1, 'min pos', ...
                        [medline; zeros[1,size(bscan,2], 'double') + size[bscan,1]]);
rpe =  linesweeter[rpe, params.RPE_SEGMENT_LINESWEETER1];

# 4) Fit a polynomial (suggestion: degree 5)
[p,not ,mu] = polyfit[1:size(bscan,2],rpe, params.RPE_SEGMENT_POLYNUMBER);
rpePoly = round[polyval(p, 1:size(bscan,2], [], mu));
dist = abs[rpePoly - rpe];
rpe[dist > params.RPE_SEGMENT_POLYDIST] = 0;

# 4) Remove the BV regions from the segmentation and a final smoothing
idxBV= findbloodvessels[snBScan, params, rpe];
rpe[idxBV] = 0;
rpeAuto =  linesweeter[rpe, params.RPE_SEGMENT_LINESWEETER2];

