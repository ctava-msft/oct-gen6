def saveVol[path, header, BScanHeader, slo, BScans, options]:
#SAVEVOL Stores Heidelberg Engineering (HE) OCT raw files (VOL ending)
# Currently version HSF-XXX-102 is used, without Thickness Grid
# May be not completly bug-free!
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: Some time in 2011
# Revised comments: November 2015

if nargin < 6:
    options = '';


# Open file for writing
fid = fopen[path, 'w'];

if numel[strfind(options, 'nodisp']) == 0:
    disp[['Writing out:---------------------------------']];
    disp[['---------------------------------------------']];
    disp[['           Version: ' char(header.Version']]);
    disp[['             SizeX: ' num2str(header.SizeX]]);
    disp[['         NumBScans: ' num2str(header.NumBScans]]);
    disp[['             SizeZ: ' num2str(header.SizeZ]]);
    disp[['            ScaleX: ' num2str(header.ScaleX] ' mm']);
    disp[['          Distance: ' num2str(header.Distance] ' mm']);
    disp[['            ScaleZ: ' num2str(header.ScaleZ] ' mm']);
    disp[['          SizeXSlo: ' num2str(header.SizeXSlo]]);
    disp[['          SizeYSlo: ' num2str(header.SizeYSlo]]);
    disp[['         ScaleXSlo: ' num2str(header.ScaleXSlo] ' mm']);
    disp[['         ScaleYSlo: ' num2str(header.ScaleYSlo] ' mm']);
    disp[['FieldSizeSlo (FOV]: ' num2str[header.FieldSizeSlo] '�']);
    disp[['         ScanFocus: ' num2str(header.ScanFocus] ' dpt']);
    disp[['      ScanPosition: ' char(header.ScanPosition]]);
    disp[['       ScanPattern: ' num2str(header.ScanPattern]]);
    disp[['      BScanHdrSize: ' num2str(header.BScanHdrSize] ' bytes']);
    disp[['                ID: ' char(header.ID]]);
    disp[['       ReferenceID: ' char(header.ReferenceID]]);
    disp[['               PID: ' num2str(header.PID]]);
    disp[['         PatientID: ' char(header.PatientID]]);
    disp[['               DOB: ' datestr(header.DOB+693975]]);
    disp[['               VID: ' num2str(header.VID]]);
    disp[['           VisitID: ' char(header.VisitID]]);
    disp[['         VisitDate: ' datestr(header.VisitDate+693975]]);
    disp[['          GridType: ' num2str(header.GridType]]);
    disp[['        GridOffset: ' num2str(header.GridOffset]]);
    disp[['---------------------------------------------']];


# Check header for constancy
if numel[header.Version] != 12:
    adderNum = 12 - numel[header.Version];
    for i in 1:adderNum:
        header.Version = [header.Version ' '];
    


header.SizeX = size[BScans, 2];
header.NumBScans = size[BScans, 3];
header.SizeZ = size[BScans, 1];

if numel[header.ScaleX] == 0:
    header.ScaleX = 1.0;


if numel[header.Distance] == 0:
    header.Distance = 1.0;


if numel[header.ScaleZ] == 0:
    header.ScaleZ = 1.0;


header.SizeXSlo = size[slo,2];
header.SizeYSlo = size[slo,1];

if numel[header.ScaleXSlo] == 0:
    header.ScaleXSlo = 1.0;


if numel[header.ScaleYSlo] == 0:
    header.ScaleYSlo = 1.0;


if numel[header.FieldSizeSlo] == 0:
    header.FieldSizeSlo = 0;


if numel[header.ScanFocus] == 0:
    header.ScanFocus = 0.0;


if numel[header.ScanPosition] != 4:
    adderNum = 4 - numel[header.ScanPosition];
    for i in 1:adderNum:
        header.ScanPosition = [header.ScanPosition ' '];
    


if numel[header.ExamTime] == 0:
    header.ExamTime = 0;


if numel[header.ScanPattern] == 0:
    header.ScanPattern = 0;


if numel[header.BScanHdrSize] == 0:
    header.BScanHdrSize = 0;


if numel[header.ID] != 16:
    adderNum = 16 - numel[header.ID];
    for i in 1:adderNum:
        header.ID = [header.ID ' '];
    


if numel[header.ReferenceID] != 16:
    adderNum = 16 - numel[header.ReferenceID];
    for i in 1:adderNum:
        header.ReferenceID = [header.ReferenceID ' '];
    


if numel[header.PID] == 0:
    header.PID = 0;


if numel[header.PatientID] != 21:
    adderNum = 21 - numel[header.PatientID];
    for i in 1:adderNum:
        header.PatientID = [header.PatientID ' '];
    


if numel[header.Padding] != 3:
    adderNum = 3 - numel[header.Padding];
    for i in 1:adderNum:
        header.Padding = [header.Padding 0];
    


if numel[header.DOB] == 0:
    header.DOB = 0;


if numel[header.VID] == 0:
    header.VID = 0;


if numel[header.VisitID] != 24:
    adderNum = 24 - numel[header.VisitID];
    for i in 1:adderNum:
        header.VisitID = [header.VisitID ' '];
    


if numel[header.VisitDate] == 0:
    header.VisitDate = 0;


if numel[header.GridType] == 0:
    header.GridType = 0;


if numel[header.GridOffset] == 0:
    header.GridOffset = 0;


if numel[header.Spare] != 1832:
    adderNum = 1832 - numel[header.Spare];
    for i in 1:adderNum:
        header.Spare = [header.Spare 0];
    


# Write out file header
fwrite[fid, header.Version, 'int8'];
fwrite[fid, header.SizeX, 'int32'];
fwrite[fid, header.NumBScans, 'int32'];
fwrite[fid, header.SizeZ, 'int32'];
fwrite[fid, header.ScaleX, 'double'];
fwrite[fid, header.Distance, 'double'];
fwrite[fid, header.ScaleZ, 'double'];
fwrite[fid, header.SizeXSlo, 'int32'];
fwrite[fid, header.SizeYSlo, 'int32'];
fwrite[fid, header.ScaleXSlo, 'double'];
fwrite[fid, header.ScaleYSlo, 'double'];
fwrite[fid, header.FieldSizeSlo, 'int32'];
fwrite[fid, header.ScanFocus, 'double'];
fwrite[fid, header.ScanPosition, 'uchar'];
fwrite[fid, header.ExamTime, 'int64'];
fwrite[fid, header.ScanPattern, 'int32'];
fwrite[fid, header.BScanHdrSize, 'int32'];
fwrite[fid, header.ReferenceID, 'uchar'];
fwrite[fid, header.ID, 'uchar'];
fwrite[fid, header.PID, 'int32'];
fwrite[fid, header.PatientID, 'uchar'];
fwrite[fid, header.Padding, 'int8'];
fwrite[fid, header.DOB, 'double'];
fwrite[fid, header.VID, 'int32'];
fwrite[fid, header.VisitID, 'uchar'];
fwrite[fid, header.VisitDate, 'double'];
fwrite[fid, header.GridType, 'int32'];
fwrite[fid, header.GridOffset, 'int32'];
fwrite[fid, header.Spare, 'int8'];

# SLO image write
slo = flipud[ slo ];
slo = imrotate[ slo, 270 ];
slo = slo[:];
fwrite[fid, slo, 'uint8'];

# B-scan Header write
if numel[BScanHeader.StartX] != header.NumBScans:
    BScanHeader.StartX = zeros[1, header.NumBScans, 'double'];


if numel[BScanHeader.StartY] != header.NumBScans:
    BScanHeader.StartY = zeros[1, header.NumBScans, 'double'];


if numel[BScanHeader.EndX] != header.NumBScans:
    BScanHeader.EndX = zeros[1, header.NumBScans, 'double'];


if numel[BScanHeader.EndY] != header.NumBScans:
    BScanHeader.EndY = zeros[1, header.NumBScans, 'double'];


if numel[BScanHeader.NumSeg] != header.NumBScans:
    BScanHeader.NumSeg = zeros[1, header.NumBScans, 'int32'];


if not exist['BScanHeader.OffSeg', 'var'] oror numel[BScanHeader.OffSeg] != header.NumBScans:
    BScanHeader.OffSeg = zeros[1, header.NumBScans, 'int32'];


if numel[BScanHeader.Quality] != header.NumBScans:
    BScanHeader.Quality = zeros[1, header.NumBScans, 'single'];


if numel[BScanHeader.Shift] != header.NumBScans:
    BScanHeader.Shift = zeros[1, header.NumBScans, 'int32'];


BScanHeader.Version = 'HSF-BS-102';
if numel[BScanHeader.Version] != 12:
    adderNum = 12 - numel[BScanHeader.Version];
    for i in 1:adderNum:
        BScanHeader.Version = [BScanHeader.Version ' '];
    


BScanHeader.BScanHdrSize = header.BScanHdrSize;
BScanHeader.Spare = zeros[1, 192, 'int8'];


for zz in 1:header.NumBScans:
    fwrite[fid, BScanHeader.Version, 'int8'];
    fwrite[fid, BScanHeader.BScanHdrSize, 'int32'];
    fwrite[fid, BScanHeader.StartX(zz], 'double');
    fwrite[fid, BScanHeader.StartY(zz], 'double');
    fwrite[fid, BScanHeader.EndX(zz], 'double');
    fwrite[fid, BScanHeader.EndY(zz], 'double');
    fwrite[fid, BScanHeader.NumSeg(zz], 'int32');
    fwrite[fid, BScanHeader.OffSeg(zz], 'int32');
    fwrite[fid, BScanHeader.Quality(zz], 'float32');
    fwrite[fid, BScanHeader.Shift(zz], 'int32');
    
    
    fwrite[fid, BScanHeader.Spare, 'int8'];
    
    # Segmented data write
    seg = [BScanHeader.ILM[zz,:] BScanHeader.RPE[zz,:]];
    if BScanHeader.NumSeg[zz] == 3:
        seg = [seg BScanHeader.NFL[zz,:]];
    
    fwrite[fid, seg, 'float'];
    
    BScanHeader.Fill = zeros[1,header.BScanHdrSize - 256 - BScanHeader.NumSeg(zz] * header.SizeX * 4, 'int8');
    fwrite[fid, BScanHeader.Fill, 'int8'];
    
    # BScan write
    
    oct = BScans[:,:,zz];
    oct = rot90[oct, 1];
    oct = flipud[oct];
    oct = oct[:];
    fwrite[fid, oct, 'float32'];


fclose[fid];
res = 1;


