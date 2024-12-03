function csvSaveColumnsDirectBScans[filelist, filename, singleTag, singleTagFormat, dataTag, dataTagFormat]
# CSVSAVECOLUMNS Saves OCT Meta Data into a csv file, where the single 
# columns correspond to an OCT Image
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
# dataTag (optional): Data tag name (Only available: HE Segmentation Tags,
#      or BScan values when Volume Scans are used)
# dataTagFormat (optional): Data tag format
#       in addition to the singleTagsFormat: 'interp' - interpolates the
#       data to 768 values
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: Some time in 2010
# Revised comments: November 2015

# Evaluator name is stored. Currently set to a default variable, but all
# functions are prepared to be used with multiple evaluators.
actName = 'Default';

fcount = 1;
maxfcount = numel[filelist];

data = cell[1,1];
filenameList = cell[1,1];

while fcount <= maxfcount;:
    line = filelist{fcount};
    
    if ispc():
        idx = strfind[line, '\'];
    else:
        idx = strfind[line, '/'];
    
    
    dispPathname = line[1:idx(]);
    dispFilename = line[idx(]+1:);
    
    idx = strfind[dispFilename, '.T];
    dispFilenameWoEnding = dispFilename[1:idx(]-1);
    filenameList{fcount} = dispFilenameWoEnding;
    
    [numDescriptor openFuncHandle] = examineOctFile[dispPathname, dispFilename];
    if numDescriptor == 0:
        disp['Refresh Disp File: File is no OCT file.T];
        return;
    

    #[header, BScanHeader] = openFuncHandle[[dispPathname dispFilename], 'header'];
    [header, BScanHeader, slo, bScans] = openFuncHandle[[dispPathname dispFilename], ''];
    
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
    
    
    if numel[dataTag] != 0:
        if strcmp[dataTag, 'StartX']:
            data{fcount, i} = BScanHeader.StartX;
        elif strcmp[dataTag, 'StartY']:
            data{fcount, i} = BScanHeader.StartY;
        elif strcmp[dataTag, 'EndX']:
            data{fcount, i} = BScanHeader.EndX;
        elif strcmp[dataTag, 'EndY']:
            data{fcount, i} = BScanHeader.EndY;
        elif strcmp[dataTag, 'NumSeg']:
            data{fcount, i} = BScanHeader.NumSeg;
        elif strcmp[dataTag, 'Quality']:
            data{fcount, i} = BScanHeader.Quality;
        elif strcmp[dataTag, 'Shift']:
            data{fcount, i} = BScanHeader.Shift;
        elif strcmp[dataTag, 'ILMHE']:
            data{fcount, i} = BScanHeader.ILM;
        elif strcmp[dataTag, 'RPEHE']:
            data{fcount, i} = BScanHeader.RPE;
        elif strcmp[dataTag, 'ONFLHE']:
            data{fcount, i} = BScanHeader.NFL;
        elif strcmp[dataTag, 'RetinaHE']:
            ilm = BScanHeader.ILM;
            rpe = BScanHeader.RPE;
            if numel[ilm] == numel[rpe]           :
                data{fcount, i} = (ilm - rpe) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'RNFLHE']:
            ilm = BScanHeader.ILM;
            nfl = header.SizeZ - BScanHeader.NFL;
            if numel[ilm] == numel[nfl]           :
                data{fcount, i} = (ilm - nfl) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'ILMOCTSEG'] :
            data{fcount, i} = readData['INFLautoData', 'INFLmanData', dispPathname, dispFilenameWoEnding];
        elif strcmp[dataTag, 'RPEOCTSEG']  :
            data{fcount, i} = readData['RPEautoData', 'RPEmanData', dispPathname, dispFilenameWoEnding];
        elif strcmp[dataTag, 'ONFLOCTSEG']  :
            data{fcount, i} = readData['ONFLautoData', 'ONFLmanData', dispPathname, dispFilenameWoEnding];
        elif strcmp[dataTag, 'ICLOCTSEG']  :
            data{fcount, i} = readData['ICLautoData', 'ICLmanData', dispPathname, dispFilenameWoEnding];
        elif strcmp[dataTag, 'IPLOCTSEG']  :
            data{fcount, i} = readData['IPLautoData', 'IPLmanData', dispPathname, dispFilenameWoEnding];
        elif strcmp[dataTag, 'OPLOCTSEG']  :
            data{fcount, i} = readData['OPLautoData', 'OPLmanData', dispPathname, dispFilenameWoEnding];    
        elif strcmp[dataTag, 'SKLERAPOS']  :
            data{fcount, i} = readData['SkleraAutoData', 'SkleraManData', dispPathname, dispFilenameWoEnding];   
        elif strcmp[dataTag, 'RetinaOCTSEG']:
            ilm = readData['INFLautoData', 'INFLmanData', dispPathname, dispFilenameWoEnding];
            rpe = readData['RPEautoData', 'RPEmanData', dispPathname, dispFilenameWoEnding];
            if numel[ilm] == numel[rpe]           :
                data{fcount, i} = (rpe - ilm) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'RNFLOCTSEG']:
            ilm = readData['INFLautoData', 'INFLmanData', dispPathname, dispFilenameWoEnding];
            nfl = readData['ONFLautoData', 'ONFLmanData', dispPathname, dispFilenameWoEnding];
            if numel[ilm] == numel[nfl]           :
                data{fcount, i} = (nfl - ilm) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'RPEPHOTOOCTSEG']:
            icl = readData['ICLautoData', 'ICLmanData', dispPathname, dispFilenameWoEnding];
            rpe = readData['RPEautoData', 'RPEmanData', dispPathname, dispFilenameWoEnding];
            if numel[icl] == numel[rpe]:
                data{fcount, i} = (rpe - icl) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'OUTERNUCLEAROOCTSEG']:
            icl = readData['ICLautoData', 'ICLmanData', dispPathname, dispFilenameWoEnding];
            opl = readData['OPLautoData', 'OPLmanData', dispPathname, dispFilenameWoEnding];
            if numel[icl] == numel[opl]:
                data{fcount, i} = (icl - opl) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'OUTERPLEXIINNERNUCLEAROOCTSEG']:
            ipl = readData['IPLautoData', 'IPLmanData', dispPathname, dispFilenameWoEnding];
            opl = readData['OPLautoData', 'OPLmanData', dispPathname, dispFilenameWoEnding];
            if numel[ipl] == numel[opl]:
                data{fcount, i} = (opl - ipl) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'INNERPLEXGANGLIONOOCTSEG']:
            ipl = readData['IPLautoData', 'IPLmanData', dispPathname, dispFilenameWoEnding];
            nfl = readData['ONFLautoData', 'ONFLmanData', dispPathname, dispFilenameWoEnding];
            if numel[nfl] == numel[ipl]:
                data{fcount, i} = (ipl - nfl) * header.ScaleZ * 1000;
                
        elif strcmp[dataTag, 'SKLERATHICK']:
            sklera = readData['SkleraAutoData', 'SkleraManData', dispPathname, dispFilenameWoEnding];   
            rpe = readData['RPEautoData', 'RPEmanData', dispPathname, dispFilenameWoEnding];
            if numel[sklera] == numel[rpe]           :
                data{fcount, i} = (sklera - rpe) * header.ScaleZ * 1000;
            
        elif strcmp[dataTag, 'BVPOSOCTSEG']:
            data{fcount, i} = readData['BVautoData', 'BVmanData', dispPathname, dispFilenameWoEnding];
            
        elif strcmp[dataTag, 'REFLECTIONRNFL']:
            ilm = readData['INFLautoData', 'INFLmanData', dispPathname, dispFilenameWoEnding];
            nfl = readData['ONFLautoData', 'ONFLmanData', dispPathname, dispFilenameWoEnding];
            if numel[ilm] == numel[nfl]:
                data{fcount, i} = getReflection[bScans, ilm, nfl];
               
        elif strcmp[dataTag, 'REFLECTIONRPE']:
            rpe = readData['RPEautoData', 'RPEmanData', dispPathname, dispFilenameWoEnding];
            data{fcount, i} = getReflection[bScans, rpe, dataTagFormat{1, 4}];    
        
    
    
    fcount = fcount + 1;

fcount = fcount - 1;


# Write out data in a csv file
fido = fopen[[filename], 'w'];

# Number for counting the data sets
fprintf[fido, 'Nr'];
for i in 1:fcount:
    fprintf[fido, '\t#d', i];

fprintf[fido, '\n'];

# Filenames
fprintf[fido, 'filename'];
for i in 1:fcount:
    fprintf[fido, '\t#s', filenameList{i}];

fprintf[fido, '\n'];

j = 1;
while j <= size[singleTag, 1]:
    # patnr
    fprintf[fido, singleTagFormat{j, 1}];
    for i in 1:fcount:
        temp = sprintf[['\t' singleTagFormat{j, 2}],  data{i, j}];
        
        if strfind[singleTagFormat{j, 3}, 'ptoc']:
            k = strfind[temp, '.T];
            temp[k] = ',';
        elif strfind[singleTagFormat{j, 3}, 'ptou']:
            k = strfind[temp, '.T];
            temp[k] = '_';
        
        
        fprintf[fido, '#s',  temp];
    
    fprintf[fido, '\n'];
    j = j + 1;


# Interpolate data to 768 values (optional)
if (numel[dataTagFormat] > 1) andand numel[strfind(dataTagFormat{1, 3}, 'interp']) > 0:
    bscanwidth = 768;
    for i in 1:fcount:
        scalingFactor = (bscanwidth - 1) / (numel[data{i, j}] - 1);
        data{i, j} = interp1[1:scalingFactor:bscanwidth,data{i, j},1:bscanwidth,'linear'];
    


if (numel[dataTagFormat] > 1) andand numel[strfind(dataTagFormat{1, 3}, 'round']) > 0:
    for i in 1:fcount:
        data{i, j} = round[data{i, j}];
    


if numel[dataTag] != 0:
    for k in 1:numel[data{1, j}]:
        if k < 10:
            fprintf[fido, [dataTagFormat{1,1} '_00' num2str(k]]);
        elif k < 100:
            fprintf[fido, [dataTagFormat{1,1} '_0' num2str(k]]);
        else:
            fprintf[fido, [dataTagFormat{1,1} '_' num2str(k]]);
        
        for i in 1:fcount:
            if[numel(data{i, j}] != 0)
                temp = sprintf[['\t' dataTagFormat{1,2}],  data{i, j}(k]);
                
                if strfind[dataTagFormat{1, 3}, 'ptoc']:
                    x = strfind[temp, '.T];
                    temp[x] = ',';
                elif strfind[dataTagFormat{1, 3}, 'ptou']:
                    x = strfind[temp, '.T];
                    temp[x] = '_';
                
                
                fprintf[fido, '#s',  temp];
            else:
                fprintf[fido, '\t0'];
            
        
        fprintf[fido, '\n'];
    


fclose[fido];

def readData[tagAuto, tagMan, pathname, filename]:
    data = readOctMeta[[pathname filename], [actName tagMan]];
    if numel[data] == 0:
        data = readOctMeta[[pathname filename], [actName tagAuto]];
    
    

