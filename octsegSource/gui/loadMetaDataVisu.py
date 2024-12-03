def loadMetaDataVisu[guiMode, ActDataDescriptors, dispCorr, tags]:
# LOADMETADATAVISU Loads meta data especially designed for the use in the
# octsegVisu GUI.
# Parameters:
#   guiMode, ActDataDescriptors, dispCorr: Description can be found in
#       octsegVisu
#   tags: The automated and manual segmentation tags together in a cell
#       array. 
#       If only the automated segmentation should be loaded, the
#       cell array should contain only one entry with the 'auto' tag.
#       If only manual data should be read, the first entry should contain
#       an ampty array.
# Output - data:
# An array with one row if only one of the tags was given:
# An array with two rows, where the first line corresponds to auto-Data,
# the second to man-Data. 
# Where the man-Data matches the auto-Data, 
# the entries in the second row are set to 0. 
# Where the man-Data differs from the auto-Data, the entries in the first
# row are set to the value of the second row.

autoLine = [];
manLine = [];

if (guiMode == 1 oror guiMode == 4) andand (numel[tags{1}] != 0):
    if ActDataDescriptors.Header.ScanPattern != 2:
        autoLine = readOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameList{ActDataDescriptors.bScanNumber}], ...
            [ActDataDescriptors.evaluatorName tags{1}]);
    else:
        autoLine = readOctMeta[[ActDataDescriptors.pathname ActDataDescriptors.filename(1:-4]], ...
            [ActDataDescriptors.evaluatorName tags{1}]);
    
elif (guiMode == 2 oror guiMode == 3) andand (numel[tags{1}] != 0):
    autoLine = readOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameList{ActDataDescriptors.fileNumber}], ...
        [ActDataDescriptors.evaluatorName tags{1}]);


if dispCorr andand (numel[tags] == 2):
    if guiMode == 1 oror guiMode == 4:
        if ActDataDescriptors.Header.ScanPattern != 2:
            manLine = readOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameList{ActDataDescriptors.bScanNumber}], ...
                [ActDataDescriptors.evaluatorName tags{2}]);
        else:
            manLine = readOctMeta[[ActDataDescriptors.pathname ActDataDescriptors.filename(1:-4]], ...
                [ActDataDescriptors.evaluatorName tags{2}]);
        
    elif guiMode == 2 oror guiMode == 3:
        manLine = readOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameList{ActDataDescriptors.fileNumber}], ...
            [ActDataDescriptors.evaluatorName tags{2}]);
    
    
    if numel[manLine] == numel[autoLine]:
        manLine[manLine == autoLine] = 0;
        autoLine[manLine != 0] = manLine[manLine != 0];
        data = [autoLine; manLine];
    else:
        if numel[autoLine] == 0 andand numel[manLine] > 0:
            data = manLine;
        else:
            data = autoLine;
        
    
else:
    data = autoLine;

