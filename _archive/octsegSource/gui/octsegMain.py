function octsegMain()
# OCTSEGMAIN Main window of the OCTSEG GUI.
#
# For a description how to use OCTSEG, please refer to OCTSEG manual.
#
# The main window essentially control the segmentation flow and displayes
# what has already been segmented or corrected in a matrix. One can open
# OCT files from here and start the segmentation process. Manual
# correction, visualization and export windows are also opened from here.
#
# Note: Due to the programm structure each of the boundary segmentation
# tasks perform stand alone, only relying on information that has been
# stored in .meta files and the original OCT image. That leads to some
# tasks that are performed multiple times during the segmentatiom process
# (e.g. denoising, aligning the scan). This is unfortunate, but as speed
# was not our main goal we accept this, as it eases the readability of the
# sourcecode. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: April 2010
# Revised comments: November 2015

#disp['Starting OCTSEG main window...T];

#--------------------------------------------------------------------------
# Load GUI Constant Values
#--------------------------------------------------------------------------

octsegConstantVariables;
global PARAMETER_FILENAME;
global TABLE_HEADERS TABLE_FORMAT TABLE_EDITABLE;
global PROCESSEDFILES;
global PROCESSEDFILESANSWER;
global FILETYPE;

#--------------------------------------------------------------------------
# GUI Global Variables
#--------------------------------------------------------------------------

# guiMode: Tells in which operating mode the gui is currently running.
# Possible values are:
# 0:    Show nothing (Default)
# 1:    HE single file edit (includes volumes), works also for image lists
# 2:    HE directory batch processing. Assumes that in the directory only
#       circular or 2D linear scans are located
# 3:    Image directory batch processing. Only one image file type allowed
#       per directory.
# 4:    Image volume (list) batch processing.
guiMode = 0;

# guiLayout: Tells which GUI layout to use.
# Possible values are:
# 0: Show nothing (Default)
# 1: HE view with SLO displayed (for guiModes 1, 2, 4)
# 2: No SLO displayed (for guiMode 3)
guiLayout = 0;

# Variables with "act" prefix:
# Data that is loaded with each image, but not displayed (e.g. raw image
# data, header information, etc...)ActDataDescriptors.filename = 0;
ActDataDescriptors.Header = []; # The current OCT file header 
                # (format similar to HE-Raw data specifications)
ActDataDescriptors.BScanHeader = []; # The B-Scan header informations. 

ActDataDescriptors.pathname = 0; # Path to currently displayed OCT data

ActDataDescriptors.filenameList = []; # Filename cell array for 
            # reading the .meta files  
            # (and OCT data in case of guiMode == 2)
            # Stored without path and without ending
ActDataDescriptors.filenameWithoutEnding = []; # The current filename without ending
ActDataDescriptors.filenameEnding = '.vol'; # The current filename ending

# Evaluator name is stored. Currently set to a default variable, but all
# functions are prepared to be used with multiple evaluators.
ActDataDescriptors.evaluatorName = 'Default';

# ActData structure: Stores the raw data from the files
ActData.slo = []; # SLO raw image data
ActData.bScans = []; # OCT raw image data

# The data in the table that is shown to visualize what segmentations 
# were performed and also how they were performed.
actTableData = [];

ParamsMain = loadParameters['MAIN', PARAMETER_FILENAME];

#--------------------------------------------------------------------------
# GUI Components
#--------------------------------------------------------------------------

# Main window
hMain = figure(...
'Visible','off',...
'Position',[200,200,800,380],...
'WindowStyle', 'normal',...
'Color', 'white',...
'HandleVisibility','callback', ...
'ResizeFcn', @hResizeFcn,...
'MenuBar', 'none');
movegui[hMain,'center'];

# Menu structure
# File menu: Loading/Saving/Open Visu
hMenFile = uimenu[hMain, 'Label', 'File'];
hMenFileOpenFile = uimenu(hMenFile, ...
    'Label', 'Open File', ...
    'Callback',{@hMenFileOpenFileCallback});
hMenFileOpenDir = uimenu(hMenFile, ...
    'Label', 'Open Directory', ...
    'Callback',{@hMenFileOpenDirCallback});
hMenFileVisu = uimenu(hMenFile,...
    'Label', 'Open in Visu',...
    'Callback',{@hMenFileVisuCallback},...
    'Separator', 'On');
hMenFileQuit = uimenu(hMenFile,...
    'Label', 'Quit',...
    'Callback',{@hMenFileQuitCallback},...
    'Separator', 'On');

# Automated menu: Perform the automated segmentations
hMenAuto = uimenu[hMain, 'Label', 'Automated'];
hMenAutoOCTSEG = uimenu(hMenAuto, ...
    'Label', 'OCTSEG Meta',...
    'Callback',{@hMenAutoOCTSEGCallback});
hMenAutoRPE = uimenu(hMenAuto, ...
    'Label', 'RPE',...
    'Callback',{@hMenAutoRPECallback});
hMenAutoONH = uimenu(hMenAuto, ...
    'Label', 'Optic Nerve Head',...
    'Callback',{@hMenAutoONHCallback});
hMenAutoBV = uimenu(hMenAuto, 'Label', 'Blood Vessels',...
    'Callback',{@hMenAutoBVCallback});
hMenAutoINFL = uimenu(hMenAuto, 'Label', 'INFL',...
    'Callback',{@hMenAutoINFLCallback});
hMenAutoInnerLayers = uimenu(hMenAuto, 'Label', 'Inner Layers',...
    'Callback',{@hMenAutoInnerLayersCallback});
hMenAutoONFL = uimenu(hMenAuto, 'Label', 'ONFL',...
    'Callback',{@hMenAutoONFLCallback});
hMenAutoAll = uimenu(hMenAuto, 'Label', 'Complete', ...
    'Callback', {@hMenAutoAllCallback},...
    'Separator', 'On');

# Optimization menu: Optimize layers from multiple choices in 3D
hMenOpt = uimenu[hMain, 'Label', 'Optimization'];
hMenOptRPE = uimenu(hMenOpt, ...
    'Label', 'RPE',...
    'Callback',{@hMenOptRPECallback});
hMenOptRPEAgain = uimenu(hMenOpt, ...
    'Label', 'RPE (with ONH and BV)',...
    'Callback',{@hMenOptRPEAgainCallback});
hMenOptINFL = uimenu(hMenOpt, ...
    'Label', 'INFL',...
    'Callback',{@hMenOptINFLCallback});


# Man menu: Perform the manual segmentations/corrections
hMenMan = uimenu[hMain, 'Label', 'Manual'];
hMenManBounds = uimenu(hMenMan, 'Label', 'Correct Boundaries',...
    'Callback',{@hMenManBoundsCallback});
hMenManBV = uimenu(hMenMan, 'Label', 'Correct Blood Vessels',...
    'Callback',{@hMenManBVCallback});
hMenManONH = uimenu(hMenMan, 'Label', 'Correct Optic Nerve Head',...
    'Callback',{@hMenManONHCallback});
hMenManSklera = uimenu(hMenMan, 'Label', 'Sklera Segmentation',...          #NOTPUBLIC
    'Callback',{@hMenManSkleraCallback});                                   #NOTPUBLIC

# Export menu: CSV-export, feature export, subvolumes and evaluation
hMenEx = uimenu[hMain, 'Label', 'Export'];
hMenExCSV = uimenu(hMenEx, 'Label', 'CSV', ...
    'Callback',{@hMenExCSVCallback});
hMenExSubvolume = uimenu(hMenEx, 'Label', 'Subvolume', ...
    'Callback',{@hMenExSubvolumeCallback});
hMenExEval = uimenu(hMenEx, 'Label', 'Evaluation', ...          #NOTPUBLIC
    'Callback',{@hMenExEvalCallback});                          #NOTPUBLIC
hMenExFeatures = uimenu(hMenEx, 'Label', 'Features', ...        #NOTPUBLIC
    'Callback',{@hMenExFeaturesCallback});                      #NOTPUBLIC

# Import menu: Add data
hMenImp = uimenu[hMain, 'Label', 'Import'];
hMenImpMeta = uimenu(hMenImp, 'Label', 'Meta Data', ...
    'Callback',{@hMenImpMetaCallback});

# The help menua currently does nothing of value
hMenHelp = uimenu[hMain, 'Label', '?'];
hMenHelpInfo = uimenu(hMenHelp, 'Label', 'Info', ...
    'Callback',{@infoCallback}); # Uses outsourced callback function for 
                                 # displaying the info text.

# A short info textox for the currently marked data in the table
hInfoText = uicontrol(hMain, 'Style','text',...
'Units','pixels',...
'String','No Info',...
'BackgroundColor', 'white',...
'FontSize', 12,...
'HorizontalAlignment', 'left', ...
'Visible', 'off');

# A figure to the display the SLO image of the currently marked data in the
# table
hSlo = axes('Units','Pixels',...
'Units','pixels',...
'Parent', hMain,...
'Visible', 'off');

# The table that shows what/how segmentations were performed
hInfoTable = uitable(hMain, ...
'CellSelectionCallback', @hInfoTableSelectCallback,...
'Units','pixels',...
'Visible', 'off');


#--------------------------------------------------------------------------
# GUI Init
#--------------------------------------------------------------------------

set[hMain,'Units','pixels'];
set[hMain,'Name','OCTSEG MAIN'];

movegui[hMain,'center'];
set[hMain,'Visible','on'];

refreshMenu();

#--------------------------------------------------------------------------
# GUI Component Handlers
#--------------------------------------------------------------------------

# Menu handlers - FILE Menu
#--------------------------------------------------------------------------

# HMENFILEOPENFILECALLBACK: 
# Opens single images/B-Scans/volumes/lists
function hMenFileOpenFileCallback[not , not ]
    [filename,pathname] = uigetfile( ...
        {'*.vol;*.oct;*.list;*.pgm; *.tif; *.jpg', 'All OCT Files (*.vol, *.oct, *.list, *.pgm, *.tif, *.jpg)';...
        '*.vol', 'Heidelberg Engineering RAW-Files (*.vol)'; ...
        '*.list', 'OCT Volume Image List (*.list)'; ...
        '*.pgm; *.tif; *.jpg; *.bmp', 'Image Files (*.pgm, *.tif, *.jpg, *.bmp)'; ...
        '*.oct', 'OCTSEG RAW-Files (*.oct)'},...
        'Select OCT file');

    if isequal[filename,0]:
        disp['Open File: Chancelled.T];
        return;
    else:
        disp[['Open File: ' pathname filename]];
    

    [numDescriptor, openFuncHandle, filenameEnding] = examineOctFile[pathname, filename];
    if numDescriptor == 0:
        disp['Open File: File is no OCT file.T];
        return;
    

    disp['Open File: OCT file check OK.T];
    switchMode[1];

    ActDataDescriptors.filename = filename;
    ActDataDescriptors.pathname = pathname;
    ActDataDescriptors.filenameEnding = filenameEnding;
    
    loadDispFile();
    
    if numDescriptor == FILETYPE.LIST:
        switchMode[4];
    elseif[ActDataDescriptors.Header.ScanPattern == 2]
        ActDataDescriptors.filenameList = cell[1,1];
        ActDataDescriptors.filenameList{1,1} = filename[1:-numel(ActDataDescriptors.filenameEnding]);
        if numDescriptor == FILETYPE.IMAGE:
            switchMode[3];
        else:
            switchMode[2];
        
    
    
    
    
    createTableData();
    loadDispInfoTableContent();
    
    refreshDispComplete;



# HMENFILEOPENDIRCALLBACK:
# Opens all B-Scans/images in a directory
function hMenFileOpenDirCallback[not , not ]
    pathname = uigetdir['.T, 'Select Folder for Batch Processing'];

    if isequal[pathname,0]:
        disp['Open Directory: Chancelled.T];
        return;
    else:
        disp[['Open Directory: ' pathname]];
    

    if ispc:
        pathname = [pathname '\'];
    else:
        pathname = [pathname '/'];
    

    # Check if there are HE Raw data files in the directory:
    ActDataDescriptors.filenameEnding = '.vol';
    suggestedMode = 2;

    files = dir[[pathname '*.vol']];

    if isempty[files]:
        disp['Open Directory: Directory contains no VOL files. \nTrying to find image files instead.T];
        suggestedMode = 3;
        
        files = dir[[pathname '*.pgm']];
        ActDataDescriptors.filenameEnding = '.pgm';
        
        if isempty[files]:
            files = dir[[pathname '*.tif']];
            ActDataDescriptors.filenameEnding = '.tif';
        
        
        if isempty[files]:
            files = dir[[pathname '*.jpg']];
            ActDataDescriptors.filenameEnding = '.jpg';
        
        
         if isempty[files]:
            files = dir[[pathname '*.bmp']];
            ActDataDescriptors.filenameEnding = '.bmp';
        
    

    if isempty[files]:
        disp['Open Directory: Directory contains no OCT files or images.T];
        return;
    

    actFilenameListTemp = cell[1,1];
    for i in 1:length[files]:
        actFilenameListTemp{i,1} = (files[i].name);
        actFilenameListTemp{i,1} = actFilenameListTemp{i,1}(1:-4);
    
    [ActDataDescriptors.filenameList, not ] = sort[actFilenameListTemp];

    switchMode[suggestedMode];

    ActDataDescriptors.filename = [ActDataDescriptors.filenameList{1} ActDataDescriptors.filenameEnding];
    ActDataDescriptors.pathname = pathname;

    loadDispFile();
    
    createTableData();
    loadDispInfoTableContent();
    
    refreshDispComplete;



# HMENFILEVISUCALLBACK:
# Open the current data in the octsegVisu visualization GUI
function hMenFileVisuCallback[not , not ]
    if guiMode != 0:
        octsegVisu[ActDataDescriptors, guiMode];
    



# HMENFILEQUITCALLBACK: Main figure delete
function hMenFileQuitCallback[not , eventdata]
    delete[hMain];



# Menu handlers - AUTOMATED Menu
#--------------------------------------------------------------------------
# 
# The callback functions do simply call the respective segmentation method 
# depending on the loaded data and write the .meta data out after the 
# segmentation. 
#
# Note: The Callback functions for the various segmentations do not check
# for the prerequisites anymore. The prerequisites should be clearly seen
# from the menu structure, and thus set in the function refreshMenu()

function hMenAutoOCTSEGCallback[hObject,eventdata]
    if isempty[eventdata]:
        [status, processedList, notProcessedList] = checkFilesIfProcessed[ActDataDescriptors, 'OCTSEG'];

        if status != PROCESSEDFILES.NONE;:
            answer = processAllQuestion['Should the OCTSEG data of all files be written new or only the remaining ones?'];
            if answer == PROCESSEDFILESANSWER.CANCEL:
                return;
            elif answer == PROCESSEDFILESANSWER.ALL:
                notProcessedList = 1:numel[ActDataDescriptors.filenameList];
            elif answer == PROCESSEDFILESANSWER.REMAINING:
            else:
                return;
            
        
    else:
        notProcessedList = eventdata;
    

    forceOptionStr = 'force';
    noheaderOptionStr = 'force octseg noheader';
    
    if ParamsMain.BINARY_META == 1:
        forceOptionStr = [forceOptionStr ' binary'];
        noheaderOptionStr = [noheaderOptionStr ' binary'];
    
    
    if guiMode == 2 oror guiMode == 3:
        tableNumber = getInfoTableColumn[guiMode, 'OCTSEG'];
        for i in 1:numel[notProcessedList]:
            writeCompleteOctMeta[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]}], ...
                ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, noheaderOptionStr, ActDataDescriptors.evaluatorName);
            actTableData{notProcessedList[i], tableNumber} = true;
            refreshDispInfoTableContent();
        
    elif guiMode == 1 oror guiMode == 4:
        tableNumber = getInfoTableColumn[guiMode, 'OCTSEG'];
        writeCompleteOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameWithoutEnding], ...
            ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, forceOptionStr);
        for i in 1:numel[notProcessedList]:
            writeCompleteOctMeta[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]}], ...
                ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, noheaderOptionStr, ActDataDescriptors.evaluatorName);
            actTableData{notProcessedList[i], tableNumber} = true;
            refreshDispInfoTableContent();
        
    

    refreshMenu();
    disp['OCTSEG meta data has been written out.T];


# HMENAUTOMEDLINECALLBACK: Perform automated medline segmentation
function hMenAutoMedlineCallback[notProcessedList]    
    # Process each BScan
    if guiMode == 1 oror guiMode == 4:
        params = loadParameters['MEDLINELIN', PARAMETER_FILENAME];

        medline = segmentMedlineVolume[ActData.bScans, params];
        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('Medline', 'auto'] getMetaTag['Medline', 'autoData']}, ...
                                     {1 medline[i, :]}, i, 0, '#.1f');
            disp[['Medline of BScan ' num2str(i] ' written out.T]);
        
    elif guiMode == 2 oror guiMode == 3:
        params = loadParameters['MEDLINECIRC', PARAMETER_FILENAME];

        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['Medline auto File: File is no OCT file.T];
                return;
            
            [ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);

            medline = segmentMedlineCirc[BScans, params];
            writeMetaAutomatedHelper[{getMetaTag('Medline', 'auto'] getMetaTag['Medline', 'autoData']}, ...
                                     {1 medline}, notProcessedList[i], 0, '#.1f');
            disp[['Medline of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    


# HMENAUTORPECALLBACK: Perform automated RPE segmentation
function hMenAutoRPECallback[hObject, eventdata]  
    if isempty[eventdata]:
        [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'RPE'];
        if not goOn, return; :
    else:
        notProcessedList = eventdata;
    

    tableNumber = getInfoTableColumn[guiMode, 'RPE'];

    # Process each BScan
    if guiMode == 1 oror guiMode == 4:
        hMenAutoMedlineCallback(); 
        medline = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);
        
        params = loadParameters['RPELIN', PARAMETER_FILENAME];
        
        [rpeAuto, rpeMult] = segmentRPEVolume[ActData.bScans, params, medline];
        
        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('RPE', 'auto'] getMetaTag['RPE', 'autoData']}, ...
                                     {1 rpeAuto[i, :]}, i , tableNumber, '#.2f');
            rpeMultTemp = [];
            for n in 1:size[rpeMult, 3]:
                rpeMultTemp = [rpeMultTemp rpeMult[i,:,n]];
            
            writeMetaAutomatedHelper[{getMetaTag('RPEMULT', 'auto'] getMetaTag['RPEMULT', 'autoData']}, ...
                                     {size[rpeMult, 3] rpeMultTemp}, i , tableNumber, '#.2f');
            
            disp[['RPE of BScan ' num2str(i] ' written out.T]);
        
    elif guiMode == 2 oror guiMode == 3:
        hMenAutoMedlineCallback[notProcessedList]; 

        params = loadParameters['RPE', PARAMETER_FILENAME];
        
        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['RPE auto File: File is no OCT file.T];
                return;
            
            [ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            
            medline = readOctMetaMerged[ActDataDescriptors, getMetaTag('Medline', 'bothData'], notProcessedList[i]);
            rpeAuto = segmentRPE[BScans, params, medline];
            
            writeMetaAutomatedHelper[{getMetaTag('RPE', 'auto'] getMetaTag['RPE', 'autoData']}, ...
                                     {1 rpeAuto}, notProcessedList[i] , tableNumber, '#.2f');
                                 
            actTableData{notProcessedList[i], tableNumber} = 'a';
            disp[['RPE of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    
    
    refreshDispInfoTableContent();
    refreshMenu();


# HMENAUTOONHCALLBACK: Perform automated ONH segmentation
# Assumes that guiMode == 1
function hMenAutoONHCallback[hObject, eventdata]
    tableNumber = getInfoTableColumn[guiMode, 'ONH'];
    params = loadParameters['ONH', PARAMETER_FILENAME];
    
    statusONH = checkFilesIfProcessed[ActDataDescriptors, getMetaTag('RPE', 'auto']);
    if statusONH != PROCESSEDFILES.ALL:
        notifierText['Perform RPE segmentation before automated ONH segmentation!'];
        return;
    
    
    rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);
    
    [onhAuto onhCenter onhRadius] = segmentONHVolume[ActData.bScans, params, rpeData];
    onhCircle = createONHCircle[ActDataDescriptors, onhCenter, onhRadius]; # ONH circle with 1mm diameter
    
    for i in 1:numel[ActDataDescriptors.filenameList]:
        writeMetaAutomatedHelper[{getMetaTag('ONH', 'auto'] ...
            getMetaTag['ONH', 'autoData'] ...
            getMetaTag['ONHCenter', 'autoData'] ...
            getMetaTag['ONHRadius', 'autoData'] ...
            getMetaTag['ONHCircle', 'autoData']}, ...
            {1 onhAuto[i, :] onhCenter onhRadius onhCircle[i, :]}, i , tableNumber);
    
    
    disp[['Optic Nerve Head segmented automatically.T]];
    
    refreshDispInfoTableContent();
    refreshMenu();


# HMENAUTOINFLCALLBACK: Perform automated INFL segmentation
function hMenAutoINFLCallback[hObject, eventdata]   
    if isempty[eventdata]:
        [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'INFL'];
        if not goOn, return; :
    else:
        notProcessedList = eventdata;
    

    tableNumber = getInfoTableColumn[guiMode, 'INFL'];
    params = loadParameters['INFL', PARAMETER_FILENAME];

    if guiMode == 1 oror guiMode == 4       :
        disp['Trying to load ONH segmentation data...T];
        onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONH', 'bothData']);
        rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);
        medlineData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);
        
        [inflAuto inflMult] = segmentINFLVolume[ActData.bScans, params, onhData, rpeData, medlineData];

        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('INFL', 'auto'] getMetaTag['INFL', 'autoData']}, ...
                                     {1 inflAuto[i, :]}, i , tableNumber, '#.2f');    
                                 
            inflMultTemp = [];
            for n in 1:size[inflMult, 3]:
                inflMultTemp = [inflMultTemp inflMult[i,:,n]];
            
            writeMetaAutomatedHelper[{getMetaTag('INFLMULT', 'auto'] getMetaTag['INFLMULT', 'autoData']}, ...
                                     {size[inflMult, 3] inflMultTemp}, i , tableNumber, '#.2f');                     
                                 
            disp[['INFL of BScan ' num2str(i] ' written out.T]);
        

    elif guiMode == 2 oror guiMode == 3:
        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['INFL auto File: File is no OCT file.T];
                return;
            
            [header, BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);

            rpe = readOctMetaMerged[ActDataDescriptors, getMetaTag('RPE', 'bothData'], notProcessedList[i]);
            medline = readOctMetaMerged[ActDataDescriptors, getMetaTag('Medline', 'bothData'], notProcessedList[i]);
            
            inflAuto = segmentINFLCirc[BScans, params, rpe, medline];

            writeMetaAutomatedHelper[{getMetaTag('INFL', 'auto'] getMetaTag['INFL', 'autoData']}, ...
                                     {1 inflAuto}, notProcessedList[i] , tableNumber, '#.2f');
            disp[['INFL of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    

    refreshDispInfoTableContent();
    refreshMenu();


# HMENAUTOBVCALLBACK: Perform automated Blood Vessel segmentation
function hMenAutoBVCallback[hObject, eventdata]  
    if isempty[eventdata]:
        [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'Blood Vessels'];
        if not goOn, return; :
    else:
        notProcessedList = eventdata;
    

    tableNumber = getInfoTableColumn[guiMode, 'Blood Vessels'];
    params = loadParameters['BV', PARAMETER_FILENAME];

    if guiMode == 1 oror guiMode == 4:
        rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);
        disp['Trying to load ONH segmentation data...T];
        onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONH', 'bothData']);
        
        bvAuto = segmentBVVolume[ActData.bScans, params, onhData, rpeData];

        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('Blood Vessels', 'auto'] getMetaTag['Blood Vessels', 'autoData']}, ...
                                     {1 bvAuto[i, :]}, i , tableNumber);          
            disp[['Blood Vessels of BScan ' num2str(i] ' written out.T]);
        
    elif guiMode == 2 oror guiMode == 3:
        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['BV auto: File is no OCT file.T];
                return;
            
            [header, BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);

            rpe = readOctMetaMerged[ActDataDescriptors, getMetaTag('RPE', 'bothData'], notProcessedList[i]);
            
            bvAuto = segmentBVCirc[BScans, params, rpe];

            writeMetaAutomatedHelper[{getMetaTag('Blood Vessels', 'auto'] getMetaTag['Blood Vessels', 'autoData']}, ...
                                     {1 bvAuto}, notProcessedList[i] , tableNumber);
            disp[['Blood Vessels of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    

    refreshDispInfoTableContent();
    refreshMenu();


# HMENAUTOINNERLAYERSCALLBACK: Perform automated Inner Layers segmentation
function hMenAutoInnerLayersCallback[hObject, eventdata]  
    if isempty[eventdata]:
        [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'Inner Layers'];
        if not goOn, return; :
    else:
        notProcessedList = eventdata;
    
    
    tableNumber = getInfoTableColumn[guiMode, 'Inner Layers'];

    if guiMode == 1 oror guiMode == 4:
        params = loadParameters['INNERLIN', PARAMETER_FILENAME];
        
        disp['Trying to load ONH segmentation data...T];
        onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONH', 'bothData']);
        rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);
        inflData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('INFL', 'bothData']);
        medlineData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);
        bvData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Blood Vessels', 'bothData']);
        
        [icl opl ipl] = segmentInnerLayersVolume[ActData.bScans, params, onhData, rpeData, inflData, medlineData, bvData];

        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('Inner Layers', 'auto'] ...
                                      getMetaTag['ICL', 'autoData'] ...
                                      getMetaTag['OPL', 'autoData'] ...
                                      getMetaTag['IPL', 'autoData']}, ...
                                     {1 icl[i, :] opl[i, :] ipl[i, :]}, ...
                                      i , tableNumber,  '#.2f');
        
    elif guiMode == 2 oror guiMode == 3:
        params = loadParameters['INNERCIRC', PARAMETER_FILENAME];

        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['Inner Layers Auto: File is no OCT file.T];
                return;
            
            [header, BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);

            rpe = readOctMetaMerged[ActDataDescriptors, getMetaTag('RPE', 'bothData'], notProcessedList[i]);
            infl = readOctMetaMerged[ActDataDescriptors, getMetaTag('INFL', 'bothData'], notProcessedList[i]);
            medline = readOctMetaMerged[ActDataDescriptors, getMetaTag('Medline', 'bothData'], notProcessedList[i]);
            bv = readOctMetaMerged[ActDataDescriptors, getMetaTag('Blood Vessels', 'bothData'], notProcessedList[i]);

            [icl opl ipl] = segmentInnerLayersCirc[BScans, params, rpe, infl, medline, bv];

            writeMetaAutomatedHelper[{getMetaTag('Inner Layers', 'auto'] ...
                                      getMetaTag['ICL', 'autoData'] ...
                                      getMetaTag['OPL', 'autoData'] ...
                                      getMetaTag['IPL', 'autoData']}, ...
                                     {1 icl opl ipl}, notProcessedList[i] , tableNumber, '#.2f');
            disp[['Inner Layers of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    

    refreshDispInfoTableContent();
    refreshMenu();



# HMENAUTOONFLCALLBACK: Perform automated ONFL segmentation
function hMenAutoONFLCallback[hObject, eventdata]
    if isempty[eventdata]:
        [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'ONFL'];
        if not goOn, return; :
    else:
        notProcessedList = eventdata;
    

    tableNumber = getInfoTableColumn[guiMode, 'ONFL'];

    if guiMode == 1 oror guiMode == 4:
        params = loadParameters['ONFLLIN', PARAMETER_FILENAME];
        
        disp['Trying to load ONH segmentation data...T];
        onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONH', 'bothData']);
        rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);
        iclData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ICL', 'bothData']);
        iplData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('IPL', 'bothData']);
        inflData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('INFL', 'bothData']);
        bvData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Blood Vessels', 'bothData']);

        [onflAuto additional] = segmentONFLVolume[ActData.bScans, params, onhData, rpeData, iclData, iplData,  inflData, bvData];

        for i in 1:numel[ActDataDescriptors.filenameList]:
            writeMetaAutomatedHelper[{getMetaTag('ONFL', 'auto'] getMetaTag['ONFL', 'autoData'] 'AdditionalInfo1'}, ...
                {1 onflAuto[i,:] additional[i,:]}, i, tableNumber, '#.2f');
        

    elif guiMode == 2 oror guiMode == 3:
        for i in 1:numel[notProcessedList]:
            [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, [ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);
            if numDescriptor == 0:
                disp['ONFL auto File: File is no OCT file.T];
                return;
            
            [header, BScanHeader, slo, BScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filenameList{notProcessedList(i]} ActDataDescriptors.filenameEnding]);

            params = loadParameters['ONFLCIRC', PARAMETER_FILENAME];

            rpe = readOctMetaMerged[ActDataDescriptors, getMetaTag('RPE', 'bothData'], notProcessedList[i]);
            icl = readOctMetaMerged[ActDataDescriptors, getMetaTag('ICL', 'bothData'], notProcessedList[i]);
            ipl = readOctMetaMerged[ActDataDescriptors, getMetaTag('IPL', 'bothData'], notProcessedList[i]);
            infl = readOctMetaMerged[ActDataDescriptors, getMetaTag('INFL', 'bothData'], notProcessedList[i]);
            bv = readOctMetaMerged[ActDataDescriptors, getMetaTag('Blood Vessels', 'bothData'], notProcessedList[i]);

            [onflAuto additionalInfo]= segmentONFLCirc[BScans, params, rpe, icl, ipl, infl, bv];

            writeMetaAutomatedHelper[{getMetaTag('ONFL', 'auto'] getMetaTag['ONFL', 'autoData'] 'AdditionalInfo1'}, ...
                {1 onflAuto additionalInfo[1,:]}, notProcessedList[i] , tableNumber, '#.2f');
            disp[['ONFL of BScan ' ActDataDescriptors.filenameList{notProcessedList(i]} ' segmented automatically.T]);
        
    

    refreshDispInfoTableContent();



# HMENAUTOALLCALLBACK: Perform all possible segmentations
function hMenAutoAllCallback[hObject, eventdata]
    [goOn notProcessedList] = getFileNumbersForAutomatedProcessing[ActDataDescriptors, guiMode, 'ONFL'];
    if not goOn, return; :

    if guiMode == 1:
        hMenAutoOCTSEGCallback[hObject, notProcessedList];
        hMenAutoRPECallback[hObject, notProcessedList];
        hMenAutoONHCallback[hObject, notProcessedList];
        hMenAutoINFLCallback[hObject, notProcessedList];
        hMenAutoBVCallback[hObject, notProcessedList];
        hMenAutoInnerLayersCallback[hObject, notProcessedList]
        hMenAutoONFLCallback[hObject, notProcessedList]
    elif guiMode == 2 oror guiMode == 3:
        hMenAutoOCTSEGCallback[hObject, notProcessedList];
        hMenAutoRPECallback[hObject, notProcessedList];
        hMenAutoINFLCallback[hObject, notProcessedList];
        hMenAutoBVCallback[hObject, notProcessedList];
        hMenAutoInnerLayersCallback[hObject, notProcessedList]
        hMenAutoONFLCallback[hObject, notProcessedList]
    

    refreshMenu();
    disp['All the segmentation data has been written out.T];


# WRITEMETAAUTOMATEDHELPER:
# Helper function for writing out the meta data results of the automated
# segmentations. 
# tag: cell array with the tags to be written out
# data: cell array with the data to be written out
#       same ordering as the associated tags
# fileNumber: filenumber of the current filenameList entry
# tableNumber: The current table column
function writeMetaAutomatedHelper[tags, data, fileNumber, tableNumber, typemode]
    if nargin < 5:
        typemode = [];
    
    
    for i in 1:numel[tags]:
        writeOctMeta([ActDataDescriptors.pathname ActDataDescriptors.filenameList{fileNumber}], ...
            [ActDataDescriptors.evaluatorName tags{i}], data{i}, typemode);
    

    if nargin >= 4 andand tableNumber > 0:
        actTableData{fileNumber, tableNumber} = 'a';
    


# Menu handlers - OPTIMIZATION Menu
#--------------------------------------------------------------------------
#
# All callback function assume that guimode == 1

# HMENOPTRPECALLBACK: Perform RPE segmentation automatization
function hMenOptRPECallback[hObject, eventdata]
    tableNumber = getInfoTableColumn[guiMode, 'RPE'];

    rpeMultNumber = readOctMeta[ActDataDescriptors.filenameList{1}, [ActDataDescriptors.evaluatorName getMetaTag('RPEMULT', 'auto']]);
    rpeMult = readOctMetaVolumeMult[ActDataDescriptors, getMetaTag('RPEMULT', 'auto'], rpeMultNumber);

    medlineData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);

    [rpeAuto medlineData] = optimizeRPE[ActDataDescriptors, rpeMult, medlineData];

    for i in 1:numel[ActDataDescriptors.filenameList]:
        writeMetaAutomatedHelper[{getMetaTag('RPE', 'auto'] getMetaTag['RPE', 'autoData'] ...
            getMetaTag['Medline', 'auto'] getMetaTag['Medline', 'autoData']}, ...
            {1 rpeAuto[i, :] 1 medlineData[i,:]}, i , tableNumber);
    

    disp[['RPE optimized and written out.T]];


function hMenOptRPEAgainCallback[hObject, eventdata]
    tableNumber = getInfoTableColumn[guiMode, 'RPE'];

    rpeMultNumber = readOctMeta[ActDataDescriptors.filenameList{1}, [ActDataDescriptors.evaluatorName getMetaTag('RPEMULT', 'auto']]);
    rpeMult = readOctMetaVolumeMult[ActDataDescriptors, getMetaTag('RPEMULT', 'auto'], rpeMultNumber);

    disp['Trying to load ONH segmentation data...T];
    onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONHCircle', 'bothData']);
    bvData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Blood Vessels', 'bothData']);

    medlineData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);

    [rpeAuto medlineData]  = optimizeRPEAgain[ActDataDescriptors, rpeMult, onhData, bvData, medlineData];

    for i in 1:numel[ActDataDescriptors.filenameList]:
        writeMetaAutomatedHelper[{getMetaTag('RPE', 'auto'] getMetaTag['RPE', 'autoData']...
            getMetaTag['Medline', 'auto'] getMetaTag['Medline', 'autoData']}, ...
            {1 rpeAuto[i, :] 1 medlineData[i,:]}, i , tableNumber);
    

    disp[['RPE optimized and written out.T]];


function hMenOptINFLCallback[not , not ]
    tableNumber = getInfoTableColumn[guiMode, 'INFL'];

    inflMultNumber = readOctMeta[ActDataDescriptors.filenameList{1}, [ActDataDescriptors.evaluatorName getMetaTag('INFLMULT', 'auto']]);
    inflMult = readOctMetaVolumeMult[ActDataDescriptors, getMetaTag('INFLMULT', 'auto'], inflMultNumber);

    disp['Trying to load ONH segmentation data...T];
    onhData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('ONHCircle', 'bothData']);
    medlineData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('Medline', 'bothData']);
    rpeData = readOctMetaMergedVolume[ActDataDescriptors, getMetaTag('RPE', 'bothData']);

    [inflAuto, rpeData, medlineData] = optimizeINFL[ActDataDescriptors, inflMult, onhData, rpeData, medlineData];

    for i in 1:numel[ActDataDescriptors.filenameList]:
        writeMetaAutomatedHelper[{getMetaTag('INFL', 'auto'] getMetaTag['INFL', 'autoData'] ...
            getMetaTag['Medline', 'auto'] getMetaTag['Medline', 'autoData'] ...
            getMetaTag['RPE', 'auto'] getMetaTag['RPE', 'autoData']}, ...
            {1 inflAuto[i, :] 1 medlineData[i,:] 1 rpeData[i,:]}, i , tableNumber);
    

    disp[['INFL optimized and written out.T]];



# Menu handlers - MANUAL Menu
#--------------------------------------------------------------------------

function hMenManBoundsCallback[not , eventdata]
    segManCorrect[ActDataDescriptors, guiMode];

    loadDispInfoTableContent()
    refreshDispInfoTableContent();
    disp[['Boundaries corrected manually.T]];
  


function hMenManBVCallback[hObject, eventdata]
    segManBV[ActDataDescriptors, guiMode];

    loadDispInfoTableContent()
    refreshDispInfoTableContent();
    disp[['Blood vessels corrected manually.T]];


function hMenManONHCallback[hObject, eventdata]
    segManONHCircle[ActDataDescriptors, guiMode];

    loadDispInfoTableContent()
    refreshDispInfoTableContent();
    disp[['ONH circle corrected manually.T]];


function hMenManSkleraCallback[hObject, eventdata]   
    segManSklera[ActDataDescriptors, guiMode];

    loadDispInfoTableContent()
    refreshDispInfoTableContent();
    disp[['Sklera segmented manually.T]];


# Menu handlers - Import Menu
#--------------------------------------------------------------------------

function hMenImpMetaCallback[hObject, eventdata]
    if guiMode == 2 oror guiMode == 3:
        importMetaBScans[ActDataDescriptors];
    


# Menu handlers - EXPORT Menu
#--------------------------------------------------------------------------

function hMenExCSVCallback[hObject, eventdata]
    if guiMode == 2 oror guiMode == 3:
        csvSaveBScans[ActDataDescriptors];
    elif guiMode == 1 oror guiMode == 4:
        csvSaveVolume[ActDataDescriptors];
    


function hMenExSubvolumeCallback[hObject, eventdata]
    if guiMode == 2 oror guiMode == 3:
    elif guiMode == 1 oror guiMode == 4:
        exportSubvolume[ActDataDescriptors]
    


function hMenExEvalCallback[hObject, eventdata]
    if guiMode == 2 oror guiMode == 3:
    elif guiMode == 1 oror guiMode == 4:
        evaluateVolume[ActDataDescriptors];
    


function hMenExFeaturesCallback[hObject, eventdata]
    if guiMode == 2 oror guiMode == 3:
        exportFeatures[ActDataDescriptors];
    


# Window control handlers
#--------------------------------------------------------------------------

function hResizeFcn[hObject, eventdata]
    refreshLayout();


# Other control handlers
#--------------------------------------------------------------------------

function hInfoTableSelectCallback[hObject, eventdata]
    newField = eventdata.Indices;

    if numel[newField] == 2 andand guiMode == 2:
        newRow = newField[1];
        ActDataDescriptors.filename = [ActDataDescriptors.filenameList{newRow} ActDataDescriptors.filenameEnding];
        loadDispFile();
    

    refreshDispComplete;


#--------------------------------------------------------------------------
# Functionality
#--------------------------------------------------------------------------

function switchMode[newMode]
    if newMode == 0:
        disp['Switching to default mode.T];
        guiMode = 0;
        guiLayout = 0;
    elif newMode == 1:
        disp['Switching to HE single file mode.T];
        guiMode = 1;
        guiLayout = 1;
    elif newMode == 2:
        disp['Switching to HE directory batch processing mode.T];
        guiMode = 2;
        guiLayout = 1;
    elif newMode == 3:
        disp['Switching to image directory batch processing mode.T];
        guiMode = 3;
        guiLayout = 2;
    elif newMode == 4:
        disp['Switching to image volume (list] batch processing mode.T);
        guiMode = 4;
        guiLayout = 1;
    else:
        disp['Mode does not exist. Switching to default mode.T];
        guiMode = 0;
        guiLayout = 0;
    

    refreshMenu();
    refreshLayout();


# REFRESHLAYOUT: Paints the Layout given by the guiLayout variable. 
# Sets all sizes and positions of buttons, figures etc...
# As a reference, the following measures are used:
#   width and height of the image window 
#   width and height of SLO images
# => this function should be called every time the window changes or other
# image data is loaded. 
# Exception: Moving through a volume does not change any sizes.
function refreshLayout()
    if guiLayout == 0:
        set[hInfoText, 'Visible', 'off'];
        set[hSlo, 'Visible', 'off'];
        set[hInfoTable, 'Visible', 'off'];
    elif guiLayout == 1:
        fpos = get[hMain, 'position'];
        width = fpos[3];
        height = fpos[4];
        border = 5;

        if width >= (900 + 3 * border):
            slowidth = 300;
        else:
            slowidth = round[(width - 3 * border] / 3);
        

        set[hInfoText, 'Position', [border border slowidth (height - 3 * border - slowidth]]);
        set[hSlo, 'Position', [border (height - slowidth - border]  slowidth slowidth]);
        set[hInfoTable, 'Position', [(2 * border + slowidth] 0 (width - 3 * border - slowidth) height]);

        set[hInfoText, 'Visible', 'on'];
        set[hSlo, 'Visible', 'on'];
        set[hInfoTable, 'Visible', 'on'];
    elif guiLayout == 2:
        fpos = get[hMain, 'position'];
        width = fpos[3];
        height = fpos[4];
        border = 5;

        set[hSlo, 'Visible', 'off'];

        set[hInfoTable, 'Position', [border 0 (width - 2 * border] (height - border)]);

        set[hInfoText, 'Visible', 'off'];
        set[hInfoTable, 'Visible', 'on'];
    

    # disp[['Layout refreshed to ' num2str(guiLayout]]);


# LOADDISPFILE: Loads a OCT dataset, and creates the table
function loadDispFile()
    [numDescriptor openFuncHandle] = examineOctFile[ActDataDescriptors.pathname, ActDataDescriptors.filename];
    if numDescriptor == 0:
        disp['Refresh Disp File: File is no OCT file.T];
        return;
    

    [ActDataDescriptors.Header, ActDataDescriptors.BScanHeader, ActData.slo, ActData.bScans] = openFuncHandle[[ActDataDescriptors.pathname ActDataDescriptors.filename]];
    
    if guiMode == 1 oror guiMode == 4:
        [ActDataDescriptors.filenameList ActDataDescriptors.filenameWithoutEnding] = createFileNameList[ActDataDescriptors.filename, ActDataDescriptors.pathname, ActDataDescriptors.Header];
    

    disp[['Display file loaded: ' ActDataDescriptors.pathname ActDataDescriptors.filename]];


# LOADDISPINFOTABLECONTENT: Reloads all the data from the meta files and
# sets the info table entries accordingly
function loadDispInfoTableContent()
    # Check OCTSEG entry
    [status, processedList, notProcessedList] = checkFilesIfProcessed[ActDataDescriptors, 'OCTSEG'];
    for k in 1:numel[processedList]:
        actTableData{processedList[k], 2} = true;
    
    for k in 1:numel[notProcessedList]:
        actTableData{notProcessedList[k], 2} = false;
    

    # Check all others
    for i in 3:numel[TABLE_HEADERS{guiMode}]:
        # First check set all to '' then check auto, then manual
        for k in 1:size[actTableData,1]:
            actTableData{k, i} = '';
        

        [status, processedListAuto] = checkFilesIfProcessed(ActDataDescriptors, ...
                                            getMetaTag[TABLE_HEADERS{guiMode}{i}, 'auto']);
            
        for k in 1:numel[processedListAuto]:
            actTableData{processedListAuto[k], i} = 'a';
        

        [status, processedListMan] = checkFilesIfProcessed(ActDataDescriptors, ...
                                            getMetaTag[TABLE_HEADERS{guiMode}{i}, 'man']);
            
        for k in 1:numel[processedListMan]:
            if strcmp[actTableData{processedListMan(k], i}, 'a'):
                actTableData{processedListMan[k], i} = 'a+m';
            else:
                actTableData{processedListMan[k], i} = 'm';
            
        
    

    # disp['Info table content has been loaded.T];


# Creates an info table, that only stores the BScan# or filenames.
# All OCTSEG specific entries are set to false.
function createTableData()
    # This function assumes, that an OCT File has been
    # loaded and the disp and act variables have been set
    if guiMode == 1 oror guiMode == 4:
        tablesize = ActDataDescriptors.Header.NumBScans;
    elif guiMode == 2 oror guiMode == 3:
        tablesize = numel[ActDataDescriptors.filenameList];
    

    actTableData = cell[tablesize, numel(TABLE_HEADERS{guiMode}]);

    if guiMode == 1 oror guiMode == 4:
        for i in 1:tablesize:
            actTableData{i,1} = i;
        
    elif guiMode == 2 oror guiMode == 3:
        for i in 1:tablesize:
            actTableData{i,1} = ActDataDescriptors.filenameList{i};
        
    

    for i in 1:tablesize:
        actTableData{i,2} = false;
    

    for i in 1:tablesize:
        for k in 3:numel[TABLE_HEADERS{guiMode}]:
            actTableData{i,k} = '';
        
    

    # disp[['Info table has been created with ' num2str(tablesize] ' entries.T]);


function refreshDispSlo()
    set[hMain,'CurrentAxes',hSlo];
    if (guiMode != 3) andand (numel[ActData.slo] > 0):
        imagesc[ActData.slo];
        axis image;
        axis off;
        colormap gray;
    
    # disp['Display SLO refreshed.T];


function refreshDispInfoText()
    text = cell[1,1];
    text{1} = ['File: ' ActDataDescriptors.filename];
    text{2} = ['Pos: ' deblank[ActDataDescriptors.Header.ScanPosition]];
    text{3} = ['ID: ' deblank[ActDataDescriptors.Header.ID]];
    text{4} = ['PatientID: ' deblank[ActDataDescriptors.Header.PatientID]];
    text{5} = ['VisitID: ' deblank[ActDataDescriptors.Header.VisitID]];

    set[hInfoText, 'String', text];
    # disp['Display infotext refreshed.T];


function refreshDispInfoTable()
    set(hInfoTable,...
        'ColumnName', TABLE_HEADERS{guiMode} ,...
        'ColumnFormat', TABLE_FORMAT{guiMode},...
        'ColumnEditable', TABLE_EDITABLE{guiMode},...
        'Data', actTableData);


function refreshDispInfoTableContent()
    set(hInfoTable,...
        'Data', actTableData);


function refreshDispComplete()
    refreshDispInfoText();
    refreshDispSlo();
    refreshDispInfoTable();
    refreshMenu();
    # disp['Display completly refreshed.T];


function refreshMenu()
    set[hMenExCSV, 'Enable', 'Off'];
    set[hMenExEval, 'Enable', 'Off'];                                      #NOTPUBLIC
    set[hMenExSubvolume, 'Enable', 'Off'];
    set[hMenExFeatures, 'Enable', 'Off'];                                  #NOTPUBLIC
    
    set[hMenImpMeta, 'Enable', 'Off'];

    set[hMenManBounds, 'Enable', 'Off'];                                   #NOTPUBLIC
    set[hMenManSklera, 'Enable', 'Off'];                                   #NOTPUBLIC
    set[hMenManBV, 'Enable', 'Off'];
    set[hMenManONH, 'Enable', 'Off'];

    set[hMenAutoOCTSEG, 'Enable', 'Off'];
    set[hMenAutoONH, 'Enable', 'Off'];
    set[hMenAutoRPE, 'Enable', 'Off'];
    set[hMenAutoBV, 'Enable', 'Off'];
    set[hMenAutoInnerLayers, 'Enable', 'Off'];
    set[hMenAutoINFL, 'Enable', 'Off'];
    set[hMenAutoONFL, 'Enable', 'Off'];

    set[hMenOptRPE, 'Enable', 'Off'];
    set[hMenOptRPEAgain, 'Enable', 'Off'];
    set[hMenOptINFL, 'Enable', 'Off'];
    
    if guiMode != 0:
        set[hMenExCSV, 'Enable', 'On'];
        
        if guiMode == 2 oror guiMode == 3:
            set[hMenAutoONH, 'Visible', 'Off'];
            set[hMenManONH, 'Visible', 'Off'];
            set[hMenExEval, 'Visible', 'Off'];                             #NOTPUBLIC
            set[hMenExSubvolume, 'Visible', 'Off'];
            
            set[hMenExFeatures, 'Visible', 'On'];                          #NOTPUBLIC
            set[hMenExFeatures, 'Enable', 'On'];                           #NOTPUBLIC
            
            set[hMenOpt, 'Visible', 'Off'];
            
            set[hMenImpMeta, 'Enable', 'On'];
        else:
            set[hMenAutoONH, 'Visible', 'On'];
            set[hMenManONH, 'Visible', 'On'];
            set[hMenExSubvolume, 'Visible', 'On'];
            set[hMenExSubvolume, 'Enable', 'On'];
            
            set[hMenExFeatures, 'Visible', 'Off'];                         #NOTPUBLIC
            
            set[hMenOpt, 'Visible', 'On'];
            set[hMenExEval, 'Enable', 'On'];                               #NOTPUBLIC
        

        set[hMenAutoOCTSEG, 'Enable', 'On'];
        
        if checkFilesIfProcessed[ActDataDescriptors, 'OCTSEG'] == PROCESSEDFILES.ALL:
            set[hMenAutoRPE, 'Enable', 'On'];
            
            if checkFilesIfProcessed[ActDataDescriptors, getMetaTag('RPE', 'auto']) == PROCESSEDFILES.ALL:
                set[hMenAutoONH, 'Enable', 'On'];
                set[hMenAutoBV, 'Enable', 'On'];
                set[hMenAutoINFL, 'Enable', 'On'];
                
                set[hMenManBounds, 'Enable', 'On'];
                set[hMenManSklera, 'Enable', 'On'];                        #NOTPUBLIC
                
                set[hMenOptRPE, 'Enable', 'On'];
                set[hMenOptRPEAgain, 'Enable', 'On'];
                set[hMenOptINFL, 'Enable', 'On'];
                
                if checkFilesIfProcessed[ActDataDescriptors, getMetaTag('INFL', 'auto']) == PROCESSEDFILES.ALL andand ...:
                   checkFilesIfProcessed[ActDataDescriptors, getMetaTag('Blood Vessels', 'auto']) == PROCESSEDFILES.ALL
                    set[hMenAutoInnerLayers, 'Enable', 'On'];
                    
                    if checkFilesIfProcessed[ActDataDescriptors, 'InnerLayersAuto'] == PROCESSEDFILES.ALL:
                        set[hMenAutoONFL, 'Enable', 'On'];
                    
                
            
            
        
        
        if checkFilesIfProcessed[ActDataDescriptors, getMetaTag('Blood Vessels', 'auto']) == PROCESSEDFILES.ALL:
            set[hMenManBV, 'Enable', 'On'];
        
        
        if checkFilesIfProcessed[ActDataDescriptors, getMetaTag('ONH', 'auto']) == PROCESSEDFILES.ALL:
            set[hMenManONH, 'Enable', 'On'];
        
    

    # disp['Menu completly refreshed.T];



