def createOverlayViewsVisu[ActDataDescriptors, SloOverlay, dispCorr, params]:
# CREATEOVERLAYVIEWSVISU: Function dedicated for the use in octsegVisu.
# Loads and creates the SloOverlay data for the overlay views.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: April 2010
# Revised comments: November 2015

SloOverlayData.data = [];    
SloOverlayData.position = []; 
SloOverlayData.opacity = []; 

if ActDataDescriptors.Header.ScanPattern == 2:
    disp['Overlay views are enabled only on volumes.T];
    return;


if SloOverlay.retinaFullOn oror SloOverlay.nflThicknessOn:
    if SloOverlay.retinaFullOn:
        rpeData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('RPE','both']);
        SloOverlayData.opacity =  params.VISU_OPACITY_THICKNESSMAP;
    elif SloOverlay.nflThicknessOn:
        onflData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('ONFL','both']);
        
        opacityTemp = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('ONHCircle','both'], 1);
        
        SloOverlayData.opacity = zeros[size(opacityTemp, 1], size[opacityTemp, 2], 'single');
        if sum[sum(opacityTemp]) == 0:
            opacityTemp = opacityTemp + 1;
        else: 
            opacityTemp = 1 - opacityTemp;
        
        SloOverlayData.opacity[opacityTemp == 1] =  params.VISU_OPACITY_THICKNESSMAP;
        SloOverlayData.opacity = flipdim[SloOverlayData.opacity,1];
        
        SloOverlayData.opacity  = registerEnfaceView[SloOverlayData.opacity, ActDataDescriptors, 'linear'];
    
    
    inflData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('INFL','both']);
    
    if SloOverlay.retinaFullOn:
        SloOverlayData.data = rpeData - inflData;
    elif SloOverlay.nflThicknessOn:
        SloOverlayData.data = onflData - inflData;
    
    
    SloOverlayData.data[SloOverlayData.data < 0] = 0;
    SloOverlayData.data = flipdim[SloOverlayData.data,1];
    
    [SloOverlayData.data, SloOverlayData.position] = registerEnfaceView[SloOverlayData.data, ActDataDescriptors];
    if SloOverlay.retinaFullOn:
        SloOverlayData.data = grayToColor(SloOverlayData.data, 'threecolors', 'colors', ...
            [params.VISU_COLOR_RETINAMAP_LOW; params.VISU_COLOR_RETINAMAP_MIDDLE; params.VISU_COLOR_RETINAMAP_HIGH],...
            'cutoff', params.VISU_RANGE_RETINAMAP);
    elif SloOverlay.nflThicknessOn:
        SloOverlayData.data = grayToColor(SloOverlayData.data, 'threecolors', 'colors',...
            [params.VISU_COLOR_RNFLMAP_LOW; params.VISU_COLOR_RNFLMAP_MIDDLE; params.VISU_COLOR_RNFLMAP_HIGH],...
            'cutoff', params.VISU_RANGE_RNFLMAP);
    

