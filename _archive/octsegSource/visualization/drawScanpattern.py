def drawScanpattern[header, BScanHeader, sloOrig, mode, varargin]:
# DRAWSCANPATTERN Draws the scanpattern of a BScan (given by the header 
# and BScanheaders) to a SLO image.  
# RES = drawScanPattern[HEADER, BSCANHEADER, SLO, MODE, OPTIONS]
# RES: RGB SLO image with the scanpattern drawn into it
# HEADER: The header of the scan, according to HE and OCTSEG spezifications
# BSCANHEADER: The BScan Header information
# SLOORIG: The Slo image, in RGB format.
# MODE: Possible drawing modes are (more than one allowed)
#   - solid (circ): solid line defining the scanpath 
#   - anything else: (circ): Two lines with the scanpattern in between
#   - quadrants (circ): The quadrants are marked with lines
#   - const (circ): Constant color of the circle
#   - pattern (volume): The complete Scanpattern is drawn
#   - position (volume): Only the position of the actual BScan is drawn
#   - area (volume): The area of the scan is shown with a bounding box
# VARARGIN: Other Options and parameters
#   - Linewidth (circ and vol): Thickness of the drawed lines
#   - Colorline (circ and vol): Color of the scanpattern
#   - Colorquadrants (circ): Color of the quadrant markers
#   - BScanNum (vol): Number of the current BScan
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: Some August 2010
# Revised comments: November 2015

if nargin < 4:
    mode = 'solid';


if size[sloOrig, 3] != 3:
    sloOrig = double[sloOrig];
    sloOrig = sloOrig - min[min(sloOrig]);
    sloOrig = sloOrig ./ max[max(sloOrig]);
    sloOrig[:,:,2] = sloOrig[:,:,1];
    sloOrig[:,:,3] = sloOrig[:,:,1];


if header.ScanPattern != 2:
    res = drawScanPatternVolume[header, BScanHeader, sloOrig, mode, varargin];
else:
    res = drawScanPatternCircle[header, BScanHeader, sloOrig, mode, varargin];



#--------------------------------------------------
# CIRCLE PATTERN
#--------------------------------------------------
def drawScanPatternCircle[header, BScanHeader, sloOrig, mode, varargin]:

if nargin < 4:
    mode = 'solid';

if (not isempty[varargin] andand iscell[varargin{1}]):
    varargin = varargin{1};

lw = 2;
ColorLine = [1 0 0];
ColorQuadrants = [0 1 0];
# Read Optional Parameters
for k in 1:2:length[varargin]:
    if (strcmp[varargin{k}, 'Linewidth']):
        lw = varargin{k+1};
    elif (strcmp[varargin{k}, 'Colorline']):
        ColorLine = varargin{k+1};
    elif (strcmp[varargin{k}, 'ColorQuadrants']):
        ColorQuadrants = varargin{k+1};
    


slo = sloOrig;
sloscale = [header.ScaleXSlo header.ScaleYSlo];

BScanStartX = BScanHeader.StartX;
BScanStartY = BScanHeader.StartY;
BScanEndX = BScanHeader.EndX;
BScanEndY = BScanHeader.EndY;

circleCenter = [round[BScanEndY * (1/sloscale(2])); round[BScanEndX * (1/sloscale(1]))];
circleStart = [round[BScanStartY * (1/sloscale(2])); round[BScanStartX * (1/sloscale(1]))];
circleVec = circleCenter - circleStart;
circleRadius = sqrt[circleVec(1] * circleVec[1] + circleVec[2] * circleVec[2]);

const = 0;
drawfactor = 1;
if numel[findstr(mode, 'const']) != 0:
    const = 1;


ColorStart = [0 0 0];
if size[ColorLine, 1] == 2:
    ColorStart = ColorLine[2,:];
    ColorLine = ColorLine[1,:];



# Draw Scan Circle
if numel[findstr(mode, 'solid']) != 0:
    for cR in circleRadius - lw:0.6:circleRadius + lw:

        if findstr['OS', header.ScanPosition]:
            for angle in pi/2:0.002:pi/2+2*pi:
                if not const:
                    drawfactor = abs[angle-pi/2] / (2*pi);
                
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = (1- drawfactor) * ColorStart[1] + drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = (1- drawfactor) * ColorStart[2] + drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = (1- drawfactor) * ColorStart[3] + drawfactor * ColorLine[3];
            
        else:
            for angle in pi/2+pi:-0.002:pi/2-pi:
                if not const:
                    drawfactor = abs[angle-pi/2-pi] / (2*pi);
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = (1- drawfactor) * ColorStart[1] + drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = (1- drawfactor) * ColorStart[2] + drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = (1- drawfactor) * ColorStart[3] + drawfactor * ColorLine[3];
            
        
    
else: # With Space in between Circles
    for cR in circleRadius - lw - 1:0.4:circleRadius - lw:

        if findstr['OS', header.ScanPosition]:
            for angle in pi/2:0.002:pi/2+2*pi:
                if not const:
                    drawfactor = abs[angle-pi/2] / (2*pi);
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = (1- drawfactor) * ColorStart[1] + drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = (1- drawfactor) * ColorStart[2] + drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = (1- drawfactor) * ColorStart[3] + drawfactor * ColorLine[3];
            
        else:
            for angle in pi/2+pi:-0.002:pi/2-pi:
                if not const:
                    drawfactor = abs[angle-pi/2-pi] / (2*pi);
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = (1- drawfactor) * ColorStart[1] + drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = (1- drawfactor) * ColorStart[2] + drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = (1- drawfactor) * ColorStart[3] + drawfactor * ColorLine[3];
            
        
    

    for cR in circleRadius + lw:0.4:circleRadius + lw + 1;:

        if findstr['OS', header.ScanPosition]:
            for angle in pi/2:0.002:pi/2+2*pi:
                if not const:
                  drawfactor = abs[angle-pi/2] / (2*pi);
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = drawfactor * ColorLine[3];
            
        else:
            for angle in pi/2+pi:-0.002:pi/2-pi:
                if not const:
                    drawfactor = abs[angle-pi/2-pi] / (2*pi);
                
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 1) = drawfactor * ColorLine[1];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 2) = drawfactor * ColorLine[2];
                slo[circleCenter(1] + round[cos(angle] * cR), circleCenter[2] + round[sin(angle] * cR), 3) = drawfactor * ColorLine[3];
            
        
    



# Draw Center
for cC1 in circleCenter[1] - 1:circleCenter[1] + 1:
    for cC2 in circleCenter[2] - 1:circleCenter[2] + 1:
        cC = [cC1; cC2];
        for cR in 0:10:
            for angle in pi/4:pi/2:2*pi:
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 1) = ColorLine[1];
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 2) = ColorLine[2];
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 3) = ColorLine[3];
            
        
    


# Draw Quadrants
if numel[findstr(mode, 'quadrants']) != 0:
    for cC1 in circleCenter[1] - 1:circleCenter[1] + 1:
        for cC2 in circleCenter[2] - 1:circleCenter[2] + 1:
            cC = [cC1; cC2];
            for cR in circleRadius - 20:0.6:circleRadius + 20:
                for angle in pi/4:pi/2:2*pi:
                    slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 1) = ColorQuadrants[1];
                    slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 2) = ColorQuadrants[2];
                    slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 3) = ColorQuadrants[3];
                
            
        
    


# Draw Start
for cC1 in circleStart[1] - 1:circleStart[1] + 1:
    for cC2 in circleStart[2] - 1:circleStart[2] + 1:
        cC = [cC1; cC2];
        for cR in 0:30:
            for angle in 2*pi+pi/4:3/2*pi:4*pi:
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 1) = ColorLine[1];
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 2) = ColorLine[2];
                slo[cC(1] + round[cos(angle] * cR), cC[2] + round[sin(angle] * cR), 3) = ColorLine[3];
            
        
    




#--------------------------------------------------
# Volume PATTERN
#--------------------------------------------------
def drawScanPatternVolume[header, BScanHeader, sloOrig, mode, varargin]:

ColorLine = [1 0 0];
BScanNum = 0;
lw = 0;

# Read Optional Parameters
if (not isempty[varargin] andand iscell[varargin{1}]):
    varargin = varargin{1};

for k in 1:2:length[varargin]:
    if (strcmp[varargin{k}, 'Colorline']):
        ColorLine = varargin{k+1};
    elif (strcmp[varargin{k}, 'BScanNum']):
        BScanNum = varargin{k+1};
    elif (strcmp[varargin{k}, 'Linewidth']):
        lw = varargin{k+1};
    


slo = sloOrig;
sloscale = [header.ScaleXSlo header.ScaleYSlo];

BScanStartX = BScanHeader.StartX;
BScanStartY = BScanHeader.StartY;
BScanEndX = BScanHeader.EndX;
BScanEndY = BScanHeader.EndY;

BScanStartX = round[BScanStartX * (1/sloscale(1]));
BScanStartY = round[BScanStartY * (1/sloscale(2]));
BScanEndX = round[BScanEndX * (1/sloscale(1]));
BScanEndY = round[BScanEndY * (1/sloscale(2]));

#[BScanStartX BScanEndX BScanStartY BScanEndY] = checkScanPositionConsistency[BScanStartX, BScanEndX, BScanStartY, BScanEndY];

if BScanNum != 0 andand numel[findstr(mode, 'position']) != 0:
    BScanStartX = BScanStartX[BScanNum];
    BScanStartY = BScanStartY[BScanNum];
    BScanEndX = BScanEndX[BScanNum];
    BScanEndY = BScanEndY[BScanNum];


if numel[findstr(mode, 'area']) != 0:
    [xPoints1 yPoints1] = getPoints[BScanStartX(1], BScanStartY[1], BScanEndX[1], BScanEndY[1], lw, size[slo]);
    [xPoints2 yPoints2] = getPoints[BScanStartX(1], BScanStartY[1], BScanStartX[), BScanStartY(], lw, size[slo]);
    [xPoints3 yPoints3] = getPoints[BScanStartX(], BScanStartY[), BScanEndX(], BScanEndY[), lw, size(slo]);
    [xPoints4 yPoints4] = getPoints[BScanEndX(1], BScanEndY[1], BScanEndX[), BScanEndY(], lw, size[slo]);
    
    xPoints = [xPoints1 xPoints2 xPoints3 xPoints4];
    yPoints = [yPoints1 yPoints2 yPoints3 yPoints4];
   
    xPoints[xPoints < 1] = 1;
    xPoints[xPoints > size(slo,2]) = size[slo,2];
    yPoints[yPoints < 1] = 1;
    yPoints[yPoints > size(slo,1]) = size[slo,1];
    
    for k in 1:numel[xPoints]:
        # Don't be confused by the ordering of x and y here. Heidelberg
        # Engineeringss spezifications are just the opposite of what a
        # normal computer scientist would do.
        slo[yPoints(k], xPoints[k], 1) = ColorLine[1];
        slo[yPoints(k], xPoints[k], 2) = ColorLine[2];
        slo[yPoints(k], xPoints[k], 3) = ColorLine[3];
    
else:
    #Draw Scanlines
    for i in 1:numel[BScanStartX]:
        [xPoints yPoints] = getPoints[BScanStartX(i], BScanStartY[i], BScanEndX[i], BScanEndY[i], lw, size[slo]);
        
        for k in 1:numel[xPoints]:
            # Don't be confused by the ordering of x and y here. Heidelberg
            # Engineeringss spezifications are just the opposite of what a
            # normal computer scientist would do.
            slo[yPoints(k], xPoints[k], 1) = ColorLine[1];
            slo[yPoints(k], xPoints[k], 2) = ColorLine[2];
            slo[yPoints(k], xPoints[k], 3) = ColorLine[3];
        
    




def getPoints[BScanStartX, BScanStartY, BScanEndX, BScanEndY, lw, sizeSlo]:

# if[BScanStartX > BScanEndX]
#     temp = BScanEndX;
#     BScanEndX = BScanStartX;
#     BScanStartX = temp;
#     
#     
# 
# 
# if[BScanStartY > BScanEndY]
#     temp = BScanEndY;
#     BScanEndY = BScanStartY;
#     BScanStartY = temp;
# 

distX = abs[BScanStartX - BScanEndX];
distY = abs[BScanStartY - BScanEndY];

if distX == 0:
    yPoints = round[min([BScanStartY BScanEndY]]:1:max[[BScanStartY BScanEndY]]);
    xPoints = zeros[1,numel(yPoints], 'single') + BScanStartX;
    if lw != 0:
        xPointsTemp = [];
        yPointsTemp = [];
        for z in -lw:1:lw:
            yPointsTemp = [yPointsTemp yPoints];
            xPointsTemp = [xPointsTemp (xPoints + z)];
        
        xPoints = xPointsTemp;
        yPoints = yPointsTemp;
    
elif distY == 0:
    xPoints = round[min([BScanStartX BScanEndX]]:1:max[[BScanStartX BScanEndX]]);
    yPoints = zeros[1,numel(xPoints], 'single') + BScanStartY;
    if lw != 0:
        xPointsTemp = [];
        yPointsTemp = [];
        for z in -lw:1:lw:
            yPointsTemp = [yPointsTemp (yPoints + z)];
            xPointsTemp = [xPointsTemp xPoints];
        
        xPoints = xPointsTemp;
        yPoints = yPointsTemp;
    
else:
    stepX = 1;
    stepY = 1;
    
    if distX > distY:
        stepY = distY / distX;
    else:
        stepX = distX / distY;
    
    
    stepX = stepX * sign[BScanEndX - BScanStartX];
    stepY = stepY * sign[BScanEndY - BScanStartY];
    
    yPoints = round[BScanStartY:stepY:BScanEndY];
    xPoints = round[BScanStartX:stepX:BScanEndX];
    
    if lw != 0:
        xPointsTemp = [];
        yPointsTemp = [];
        for z in -lw:1:lw:
            yPointsTemp = [yPointsTemp (yPoints + z)];
            xPointsTemp = [xPointsTemp (xPoints + z)];
        
        xPoints = xPointsTemp;
        yPoints = yPointsTemp;
    


xPoints[xPoints < 1] = 1;
xPoints[xPoints > sizeSlo(2]) = sizeSlo[2];
yPoints[yPoints < 1] = 1;
yPoints[yPoints > sizeSlo(1]) = sizeSlo[1];

