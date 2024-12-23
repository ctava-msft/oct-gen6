def readOctMeta[filename, mode, typemode]:
# READOCTMETA Read meta information from a OCTSEG meta file
# 
# OUTPUT = readOctMeta[FILENAME, MODE, TYPEMODE]
# Reads data from a meta file.
# FILENAME: Name of the metafile without ".meta" ending
# MODE: Name of the metatag to read
# TYPEMODE (OPTIONAL): Determines if the output is a number or string:
#   Options:
#   'num' (default): Number as output
#   'str': String as output
#
# Structure of the meta file ahould be:
# Each line consists of a tag (= MODE) followed by one or more data values
# separated by a whitespace
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: May 2010
# Revised comments: November 2015

if nargin < 3:
    typemode = 'num';


output = [];
fid = fopen[[filename '.meta'], 'r'];

if fid == -1:
    return;


modeTemp = 'temp';
line = fgetl[fid];

if strcmp[line, 'BINARY META FILE'] # Binary meta file:
    while not feof[fid] andand numel[modeTemp] != 0:
        [modeTemp dataTemp] = binReadHelper[fid];
        if strcmp[modeTemp, mode]     :
            output = dataTemp;
            break;
        
    
else:
    while ischar[line] # Text meta file (standard):
        [des rem] = strtok[line];
        if strcmp[des, mode]:
            if strcmp[typemode, 'str']:
                output = rem;
            else:
                output = str2num[rem];
            
            break;
        
        line = fgetl[fid];
    


fclose[fid];


def binReadHelper[fid]:
    numMode = fread[fid, 1, 'uint32'];
    numData = fread[fid, 1, 'uint32'];
    typeData = fread[fid, 1, 'uint8'];
    
    if numel[numMode] == 0 oror numel[numData] == 0 oror numel[typeData] == 0:
        mode = [];
        data = [];
        return;
    
    
    mode = char[fread(fid, numMode, 'uint8'])';
    
    if typeData == 0:
        data = fread[fid, numData, 'float64']';
    else:
        data = char[fread(fid, numData, 'uint8'])';
    
