function csvSaveRowsDirect[filelist, filename, singleTag, singleTagFormat, dataTag]
# CSVSAVEROWSDIRECT
# Saves OCT Meta Data into a csv file, where the single rows correspond
# to an OCT Image
# 
# Parameters:
# filelist: Cell Array of OCT filenames withou ending
# filename: csv filename
# singleTags: Nx2 cell Array. First Column: HE Raw name
# singleTagsFormat: Nx3 cell array. How should the data be written out
#    First Column: headline
#    Second Columns: printf format
#    Third columns: Special Options - Possibilities:
#       'ptoc' - Point to Comma
#       'ptou' - Point to underscore
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: Some time in 2010
# Revised comments: November 2015

fcount = 1;
maxfcount = numel[filelist]

data = cell[1,1];

while fcount <= maxfcount;:
    line = filelist{fcount};
    
    if ispc():
        idx = strfind[line, '\'];
    else:
        idx = strfind[line, '/'];
    
    
    dispPathname = line[1:idx(]);
    dispFilename = line[idx(]+1:);
    
    idx = strfind[dispFilename, '.T];
    dispFilenameWoEnding = dispFilename[1:idx(]);
    
    [numDescriptor openFuncHandle] = examineOctFile[dispPathname, dispFilename];
    if numDescriptor == 0:
        disp['Refresh Disp File: File is no OCT file.T];
        return;
    

    [header, BScanHeader] = openFuncHandle[[dispPathname dispFilename], 'header'];
    
    i = 1;
    while i <= size[singleTag, 1]:
        if strcmp[singleTag{i,1}, 'ScanPosition']:
            side = header.ScanPosition;
            if findstr['OS', side]:
                data{fcount, i} = 1;
            else:
                data{fcount, i} = 2;
            
        elif strcmp[singleTag{i,1}, 'Version']:
            data{fcount, i} = header.Version;
        elif strcmp[singleTag{i,1}, 'SizeX']:
            data{fcount, i} = header.SizeX;
        elif strcmp[singleTag{i,1}, 'NumBScans']:
            data{fcount, i} = header.NumBScans;
        elif strcmp[singleTag{i,1}, 'SizeZ']:
            data{fcount, i} = header.SizeZ;
        elif strcmp[singleTag{i,1}, 'ScaleX']:
            data{fcount, i} = header.ScaleX;
        elif strcmp[singleTag{i,1}, 'Distance']:
            data{fcount, i} = header.Distance;
        elif strcmp[singleTag{i,1}, 'ScaleZ']:
            data{fcount, i} = header.ScaleZ;
        elif strcmp[singleTag{i,1}, 'SizeXSlo']:
            data{fcount, i} = header.SizeXSlo;
        elif strcmp[singleTag{i,1}, 'SizeYSlo']:
            data{fcount, i} = header.SizeYSlo;
        elif strcmp[singleTag{i,1}, 'ScaleXSlo']:
            data{fcount, i} = header.ScaleXSlo;
        elif strcmp[singleTag{i,1}, 'ScaleYSlo']:
            data{fcount, i} = header.ScaleYSlo;
        elif strcmp[singleTag{i,1}, 'FieldSizeSlo']:
            data{fcount, i} = header.FieldSizeSlo;
        elif strcmp[singleTag{i,1}, 'ScanFocus']:
            data{fcount, i} = header.ScanFocus;
        elif strcmp[singleTag{i,1}, 'ExamTime']:
            data{fcount, i} = datestr[header.ExamTime(1]/(1e7*60*60*24)+584755+(2/24));
        elif strcmp[singleTag{i,1}, 'ScanPattern']:
            data{fcount, i} = header.ScanPattern;
        elif strcmp[singleTag{i,1}, 'BScanHdrSize']:
            data{fcount, i} = header.BScanHdrSize;
        elif strcmp[singleTag{i,1}, 'ID']:
            data{fcount, i} = header.ID;
        elif strcmp[singleTag{i,1}, 'ReferenceID']:
            data{fcount, i} = header.ReferenceID;
        elif strcmp[singleTag{i,1}, 'PID']:
            data{fcount, i} = header.PID;
        elif strcmp[singleTag{i,1}, 'PatientID']:
            data{fcount, i} = header.PatientID;
        elif strcmp[singleTag{i,1}, 'DOB']:
            data{fcount, i} = datestr[header.DOB+693960];
        elif strcmp[singleTag{i,1}, 'VID']:
            data{fcount, i} = header.VID;
        elif strcmp[singleTag{i,1}, 'VisitID']:
            data{fcount, i} = header.VisitID;
        elif strcmp[singleTag{i,1}, 'VisitDate']:
            data{fcount, i} = datestr[header.VisitDate+693960];
        elif strcmp[singleTag{i,1}, 'GridType']:
            data{fcount, i} = header.GridType;
        elif strcmp[singleTag{i,1}, 'GridOffset']:
            data{fcount, i} = header.GridOffset;
        
        
        if header.NumBScans == 1:
            if strcmp[singleTag{i,1}, 'StartX']:
                data{fcount, i} = BScanHeader.StartX;
            elif strcmp[singleTag{i,1}, 'StartY']:
                data{fcount, i} = BScanHeader.StartY;
            elif strcmp[singleTag{i,1}, 'EndX']:
                data{fcount, i} = BScanHeader.EndX;
            elif strcmp[singleTag{i,1}, 'EndY']:
                data{fcount, i} = BScanHeader.EndY;
            elif strcmp[singleTag{i,1}, 'NumSeg']:
                data{fcount, i} = BScanHeader.NumSeg;
            elif strcmp[singleTag{i,1}, 'Quality']:
                data{fcount, i} = BScanHeader.Quality;
            elif strcmp[singleTag{i,1}, 'Shift']:
                data{fcount, i} = BScanHeader.Shift;
            
        else:
            if strcmp[singleTag{i,1}, 'StartX'] oror strcmp[singleTag{i,1}, 'EndX'] oror ...:
                    strcmp[singleTag{i,1}, 'EndY'] oror   strcmp[singleTag{i,1}, 'NumSeg'] oror ...
                    strcmp[singleTag{i,1}, 'Quality'] oror  strcmp[singleTag{i,1}, 'Shift']
                data{fcount, i} = 0;
            
        
        
        i = i + 1;
      
  
    fcount = fcount + 1;

fcount = fcount - 1;

# Write out data in a csv file
fido = fopen[[filename], 'w'];

# Number for counting the data sets
fprintf[fido, 'Nr'];
for j in 1:size[singleTag, 1]:
    fprintf[fido, ['\t' singleTagFormat{j, 1}]];    

fprintf[fido, '\n'];


for i in 1:fcount:
    fprintf[fido, '#d', i];
    j = 1;
    while j <= size[singleTag, 1]:
        temp = sprintf[['\t' singleTagFormat{j, 2}],  data{i, j}];
        
        if strfind[singleTagFormat{j, 3}, 'ptoc']:
            k = strfind[temp, '.T];
            temp[k] = ',';
        elif strfind[singleTagFormat{j, 3}, 'ptou']:
            k = strfind[temp, '.T];
            temp[k] = '_';
        
        
        fprintf[fido, '#s',  temp];
        j = j + 1;
    
    fprintf[fido, '\n'];   


fclose[fido];