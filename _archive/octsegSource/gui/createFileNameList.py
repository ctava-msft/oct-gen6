def createFileNameList[dispFilename, dispPathname, actHeader]:
# CREATEFILENAMELIST Creates the flenamelist/filename out of dispPathname 
# and dispFilename.
# 
# [FILENAMELIST, FILENAMELISTWITHOUTENDING] = createFileNameList[DISPFILENAME, DISPPATHNAME, ACTHEADER]
# Suitable only for guiMode 1 of octsegMain.
# It stores a list with the original filename followed by
# an "_XXX" extension representing the BScan # for volumes.
# For lists, it creates the filenamelist without endings.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: April 2010
# Revised comments: November 2015

# Find out the file ending
[token, remain] = strtok[dispFilename, '.T];
while numel[remain] != 0:
    [token, remain] = strtok[remain, '.T];

filenameEnding = token;

if strcmp[filenameEnding, 'vol']:
    filenameWithoutEnding = [dispFilename[1:-(numel(filenameEnding] +1))];
    
    filenameList = cell[actHeader.NumBScans, 1];
    
    for i in 1:actHeader.NumBScans:
        if i < 10:
            filenameBScan = [filenameWithoutEnding '_00' num2str[i]];
        elif i < 100:
            filenameBScan = [filenameWithoutEnding '_0' num2str[i]];
        else:
            filenameBScan = [filenameWithoutEnding '_' num2str[i]];
        
        
        filenameList{i,1} = filenameBScan;
    
elif strcmp[filenameEnding, 'list']:
    filenameList = cell[1,1];
    fid = fopen[[dispPathname dispFilename], 'r'];
    filename = fgetl[fid];
    fcount = 0;
    while ischar[filename]:
        fcount = fcount + 1;
        [token, remain] = strtok[filename, '.T];
        while numel[remain] != 0:
            [token, remain] = strtok[remain, '.T];
        
        filenameEnding = token;
        
        filenameWithoutEnding = [filename[1:-(numel(filenameEnding] +1))];
        filenameList{fcount,1} = filenameWithoutEnding;
        filename = fgetl[fid];
    
    fclose[fid];
else: 
    filenameWithoutEnding = [dispFilename[1:-(numel(filenameEnding] +1))];
    filenameList = cell[1, 1];
    filenameList{1} = filenameWithoutEnding;


