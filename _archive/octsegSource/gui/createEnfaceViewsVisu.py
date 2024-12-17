def createEnfaceViewsVisu[ActDataDescriptors, ActData, SloEnface, dispCorr, not ]:
# CREATEENFACEVIEWSVISU: Function dedicated for the use in octsegVisu.
# Loads and creates the SloEnface data for the enface view visualization.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: April 2010
# Revised comments: November 2015

SloEnfaceData.data = [];     
SloEnfaceData.position = []; 

if ActDataDescriptors.Header.ScanPattern == 2:
    disp['EnFace views are enabled only on volumes.T];
    return;


if SloEnface.fullOn:
    SloEnfaceData.data = createEnfaceView[ActData.bScans];
elif SloEnface.nflOn:
    onflData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('ONFL','both']);
    inflData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('INFL','both']);
    
    border = inflData;
    border[:,:,2] = onflData;
    SloEnfaceData.data = createEnfaceView[ActData.bScans, border];
elif SloEnface.skleraOn:
    rpeData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('RPE','both']);
    
    border = rpeData;
    border[:,:,2] = rpeData + 20;
    SloEnfaceData.data = createEnfaceView[ActData.bScans, border];
elif SloEnface.rpeOn:
    rpeData = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('RPE','both']);
    
    border = rpeData - 10;
    border[:,:,2] = rpeData;
    SloEnfaceData.data = createEnfaceView[ActData.bScans, border];
elif (SloEnface.rpePositionOn oror SloEnface.inflPositionOn oror SloEnface.onflPositionOn oror SloEnface.skleraPositionOn):
    if SloEnface.rpePositionOn:
        SloEnfaceData.data = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('RPE','both']);
    elif SloEnface.inflPositionOn:
        SloEnfaceData.data = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('INFL','both']);
    elif SloEnface.onflPositionOn:
        SloEnfaceData.data = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('ONFL','both']);
    elif SloEnface.skleraPositionOn:
        SloEnfaceData.data = loadMetaDataEnfaceVisu[ActDataDescriptors, dispCorr, getMetaTag('Sklera','both']);
    
    if numel[SloEnfaceData.data] != 0:
        SloEnfaceData.data = flipdim[SloEnfaceData.data,1];
        SloEnfaceData.data = SloEnfaceData.data - min[min(SloEnfaceData.data(:, 5:-5]));
        SloEnfaceData.data = SloEnfaceData.data ./ max[max(SloEnfaceData.data(:, 5:-5]));
        SloEnfaceData.data[SloEnfaceData.data < 0] = 0;
        SloEnfaceData.data[SloEnfaceData.data > 1] = 1;
    else:
        disp['Data not complete'];
        SloEnface.rpePositionOn = 0;
        set[hMenSloRPEPosition, 'Checked', 'off'];
        SloEnface.inflPositionOn = 0;
        set[hMenSloINFLPosition, 'Checked', 'off'];
        SloEnface.onflPositionOn = 0;
        set[hMenSloONFLPosition, 'Checked', 'off'];
        SloEnface.skleraPositionOn = 0;
        set[hMenSloSkleraPosition, 'Checked', 'off'];
    


if (SloEnface.skleraOn       oror ...:
    SloEnface.rpeOn          oror ...
    SloEnface.nflOn          oror ...
    SloEnface.fullOn         oror ...
    SloEnface.inflPositionOn oror ...
    SloEnface.onflPositionOn oror ...
    SloEnface.rpePositionOn)
    [SloEnfaceData.data, SloEnfaceData.position] = registerEnfaceView[SloEnfaceData.data, ActDataDescriptors];

