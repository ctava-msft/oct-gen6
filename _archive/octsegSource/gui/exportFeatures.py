function  exportFeatures[DataDescriptors]
# EXPORTFEATURES GUI wrapper for the feature export functions for
# circular B-Scans, namely generateFeaturesBScans and 
# writeFeatureFile.
# 
# Its best to have a look at the GUI to get the sources - nothing special
# here, just various option fields that are parsed and handed over to the
# export functions.
#
# Note that it is best to have a 'Diagnosis' meta data tag filled in the
# meta files. This has to be done by the import Metadata tool.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: April 2010
# Revised comments: November 2015

#--------------------------------------------------------------------------
# Variables
#--------------------------------------------------------------------------
Layers.retina = 1;
Layers.rnfl = 1;
Layers.ipl = 1;
Layers.opl = 1;
Layers.onl = 1;
Layers.rpe = 1;
Layers.bv = 1;

Additional.filename = 1;
Additional.age = 1;
Additional.patientID = 1;

Features.normalize = 1;
Features.samples = 32;
Features.onlyAuto = 0;
Features.class = 'Diagnosis';
Features.numSamplesPCA = 64;
Features.numEV = 10;
Features.pcaAllClasses = [0 1 2 3];
Features.pcaNormalClasses = 0;
Features.normFocus = 1;
Features.ageNormalize = 1;
Features.ageNormalizeSamples = 32;
Features.ageNormalizeRefAge = 50;
Features.ageNormalClass = 0;

Types.completeStd = 1;
Types.meanSections = 1;
Types.pcaAll = 1;
Types.pcaNormal = 1;

#--------------------------------------------------------------------------
# GUI Components
#--------------------------------------------------------------------------

f = figure('Visible','off','Position',[360,500,500,680],...
    'WindowStyle', 'normal',...
    'MenuBar', 'none', ...
    'ResizeFcn', {@refreshLayout},...
    'CloseRequestFcn', {@hCloseRequestFcn},...
    'Color', 'white');

#------------Checkboxes for Layers---------------
hTextLayers = uicontrol('Style','text','String','Layers',...
    'BackgroundColor', 'white',...
    'FontSize', 16,...
    'HorizontalAlignment', 'center');

hCheckRetina = uicontrol('Style','checkbox','String', 'Retina',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckRNFL = uicontrol('Style','checkbox','String', 'RNFL',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckIPLGCL = uicontrol('Style','checkbox','String', 'IPL+GCL',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckOPLINL = uicontrol('Style','checkbox','String', 'OPL+INL',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckONL = uicontrol('Style','checkbox','String', 'ONL',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckRPEPhoto = uicontrol('Style','checkbox','String', 'RPE+Photoreceptors',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

hCheckBV = uicontrol('Style','checkbox','String', 'Blood Vessels',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hLayersCallback});

#------------Additional Information---------------
hTextAdditional = uicontrol('Style','text','String','Additional Information',...
    'BackgroundColor', 'white',...
    'FontSize', 16,...
    'HorizontalAlignment', 'center');

hCheckFilename = uicontrol('Style','checkbox','String', 'Filename',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hAdditionalCallback});

hCheckAge = uicontrol('Style','checkbox','String', 'Age',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hAdditionalCallback});

hCheckPatientID = uicontrol('Style','checkbox','String', 'PatientID',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hAdditionalCallback});


#------------Features General---------------
hTextFeaturesGeneral = uicontrol('Style','text','String','General Feature Options',...
    'BackgroundColor', 'white',...
    'FontSize', 16,...
    'HorizontalAlignment', 'center');

hCheckNormalize = uicontrol('Style','checkbox','String', 'Normalize',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeaturesGeneralCallback});

hCheckAuto = uicontrol('Style','checkbox','String', 'Aut. Seg. Only',...
    'Value', 0, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeaturesGeneralCallback});

hTextSamples = uicontrol('Style','text','String','#Samples: ',...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'HorizontalAlignment', 'center', ...
    'Callback',{@hFeaturesGeneralCallback});

hEditSamples = uicontrol('Style','edit','String', '32',...
    'BackgroundColor', [1 1 1],...
    'FontSize', 10,...
    'Min', 1, 'Max', 1, ...
    'HorizontalAlignment', 'left',...
    'Units','pixels', ...
    'Callback',{@hFeaturesGeneralCallback});

hTextMeta = uicontrol('Style','text','String','Class: ',...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'HorizontalAlignment', 'center', ...
    'Callback',{@hFeaturesGeneralCallback});

hEditMeta = uicontrol('Style','edit','String', 'Diagnosis',...
    'BackgroundColor', [1 1 1],...
    'FontSize', 10,...
    'Min', 1, 'Max', 1, ...
    'HorizontalAlignment', 'left',...
    'Units','pixels', ...
    'Callback',{@hFeaturesGeneralCallback});

hTextNumEV = uicontrol('Style','text','String','#EV: ',...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'HorizontalAlignment', 'center', ...
    'Callback',{@hFeaturesGeneralCallback});

hEditNumEV = uicontrol('Style','edit','String', '10',...
    'BackgroundColor', [1 1 1],...
    'FontSize', 10,...
    'Min', 1, 'Max', 1, ...
    'HorizontalAlignment', 'left',...
    'Units','pixels', ...
    'Callback',{@hFeaturesGeneralCallback});

hCheckFocusNormalize = uicontrol('Style','checkbox','String', 'Normalize Focus',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeaturesGeneralCallback});

hCheckAgeNormalize = uicontrol('Style','checkbox','String', 'Normalize Age',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeaturesGeneralCallback});

hTextAgeSamples = uicontrol('Style','text','String','#AgeSamples: ',...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'HorizontalAlignment', 'center', ...
    'Callback',{@hFeaturesGeneralCallback});

hEditAgeSamples = uicontrol('Style','edit','String', '32',...
    'BackgroundColor', [1 1 1],...
    'FontSize', 10,...
    'Min', 1, 'Max', 1, ...
    'HorizontalAlignment', 'left',...
    'Units','pixels', ...
    'Callback',{@hFeaturesGeneralCallback});

hTextAgeNorm = uicontrol('Style','text','String','AgeNorm: ',...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'HorizontalAlignment', 'center', ...
    'Callback',{@hFeaturesGeneralCallback});

hEditAgeNorm = uicontrol('Style','edit','String', '50',...
    'BackgroundColor', [1 1 1],...
    'FontSize', 10,...
    'Min', 1, 'Max', 1, ...
    'HorizontalAlignment', 'left',...
    'Units','pixels', ...
    'Callback',{@hFeaturesGeneralCallback});



#------------Feature Types---------------
hTextFeatureTypes = uicontrol('Style','text','String','Feature Types',...
    'BackgroundColor', 'white',...
    'FontSize', 16,...
    'HorizontalAlignment', 'center');

hCheckFeatureCompleteStd = uicontrol['Style','checkbox','String', 'Complete Std (min, max, mean, median]',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeatureTypesCallback});

hCheckFeatureMeanSections = uicontrol('Style','checkbox','String', 'Mean of Sections',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeatureTypesCallback});

hCheckFeaturePCAAll = uicontrol['Style','checkbox','String', 'PCA (all] Eigenvalues',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeatureTypesCallback});

hCheckFeaturePCANormal = uicontrol['Style','checkbox','String', 'PCA (normal] Eigenvalues',...
    'Value', 1, ...
    'BackgroundColor', 'white',...
    'FontSize', 12,...
    'Callback',{@hFeatureTypesCallback});


#------------Buttons---------------
hExport = uicontrol('Style','pushbutton','String','Generate Features',...
    'Position',[10,10,780,80],...
    'FontSize', 16,...
    'Callback',{@hExportCallback});

hCancel = uicontrol('Style','pushbutton','String','Cancel',...
    'Position',[10,10,780,80],...
    'FontSize', 16,...
    'Callback',{@hCloseRequestFcn});

#--------------------------------------------------------------------------
# GUI Init
#--------------------------------------------------------------------------

set[f,'Units','pixels'];
set[f,'Name','Export Features'];

movegui[f,'center'];
set[f,'Visible','on'];

function hCloseRequestFcn[hObject, eventdata, handles]
    #uiresume[hObject];
    delete[f];


function hLayersCallback[hObject, eventdata]
    Layers.retina = get[hCheckRetina, 'Value'];
    Layers.rnfl = get[hCheckRNFL, 'Value'];
    Layers.ipl = get[hCheckIPLGCL, 'Value'];
    Layers.opl = get[hCheckOPLINL, 'Value'];
    Layers.onl = get[hCheckONL, 'Value'];
    Layers.rpe = get[hCheckRPEPhoto, 'Value'];
    Layers.bv = get[hCheckBV, 'Value'];


function hAdditionalCallback[hObject, eventdata]
    Additional.filename = get[hCheckFilename, 'Value'];
    Additional.age = get[hCheckAge, 'Value'];
    Additional.patientID = get[hCheckPatientID, 'Value'];


function hFeaturesGeneralCallback[hObject, eventdata]
    Features.normalize = get[hCheckNormalize, 'Value'];
    Features.samples = str2double[get(hEditSamples, 'String']);
    Features.onlyAuto = get[hCheckAuto, 'Value'];
    Features.class = get[hEditMeta, 'String'];
    Features.numEV = str2double[get(hEditNumEV, 'String']);
    Features.normFocus = get[hCheckFocusNormalize, 'Value'];
    
    Features.ageNormalize = get[hCheckAgeNormalize, 'Value'];
    Features.ageNormalizeSamples = str2double[get(hEditAgeSamples, 'String']);
    Features.ageNormalizeRefAge = str2double[get(hEditAgeNorm, 'String']);


function hFeatureTypesCallback[hObject, eventdata]
    Types.completeStd = get[hCheckFeatureCompleteStd, 'Value'];
    Types.meanSections = get[hCheckFeatureMeanSections, 'Value'];
    Types.pcaAll = get[hCheckFeaturePCAAll, 'Value'];
    Types.pcaNormal = get[hCheckFeaturePCANormal, 'Value'];


# REFRESHLAYOUT: Paints the Layout
function refreshLayout[hObject, eventdata]
        fpos = get[f, 'position'];
        width = fpos[3];
        height = fpos[4];
        border = 5;

        stdWidth = width - 2 * border;
        stdWidth2 = (width - 3 * border) / 2;
        stdWidth3 = (width - 4 * border) / 3;
        stdWidth4 = (width - 5 * border) / 4;
        stdHeight = 40;
        
        # First: Draw layer possibilities
        actHeight = height - stdHeight - 2 * border;
        set[hTextLayers, 'Position', [border actHeight stdWidth stdHeight]];
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckRetina, 'Position', [border actHeight stdWidth4 stdHeight]];
        set[hCheckRNFL, 'Position', [(2 * border + stdWidth4] actHeight stdWidth4 stdHeight]);
        set[hCheckIPLGCL, 'Position', [(3 * border + 2 * stdWidth4] actHeight stdWidth4 stdHeight]);
        set[hCheckOPLINL, 'Position', [(4 * border + 3 * stdWidth4] actHeight stdWidth4 stdHeight]);
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckONL, 'Position', [border actHeight stdWidth4 stdHeight]];
        set[hCheckRPEPhoto, 'Position', [(2 * border + stdWidth4] actHeight stdWidth4 stdHeight]);
        set[hCheckBV, 'Position', [(3 * border + 2 * stdWidth4] actHeight stdWidth4 stdHeight]);
        
        # Draw additional information possibilites
        actHeight = actHeight - stdHeight - 2 * border;
        set[hTextAdditional, 'Position', [border actHeight stdWidth stdHeight]];
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckFilename, 'Position', [border actHeight stdWidth3 stdHeight]];
        set[hCheckAge, 'Position', [(2 * border + stdWidth3] actHeight stdWidth3 stdHeight]);
        set[hCheckPatientID, 'Position', [(3 * border + 2 * stdWidth3] actHeight stdWidth3 stdHeight]);
        
        # Draw general feature options
        actHeight = actHeight - stdHeight - 2 * border;
        set[hTextFeaturesGeneral, 'Position', [border actHeight stdWidth stdHeight]];
        
        actHeight = actHeight - stdHeight - border;
        set[hTextMeta, 'Position', [border actHeight stdWidth3 stdHeight]];
        set[hEditMeta, 'Position', [(2 * border + stdWidth3] actHeight stdWidth3 stdHeight]);
        set[hCheckAuto, 'Position', [(3 * border + 2 * stdWidth3] actHeight stdWidth3 stdHeight]);
        
        actHeight = actHeight - stdHeight - border;   
        set[hCheckNormalize, 'Position', [border actHeight stdWidth3 stdHeight]];
        set[hCheckFocusNormalize, 'Position', [(2 * border + stdWidth3] actHeight stdWidth3 stdHeight]);

        actHeight = actHeight - stdHeight - border;
        set[hTextSamples, 'Position', [border actHeight (stdWidth3 * 0.5] stdHeight]);
        set[hEditSamples, 'Position', [(border + stdWidth3 * 0.5] actHeight (stdWidth3 * 0.5) stdHeight]);
        set[hTextNumEV, 'Position', [(2 * border + stdWidth3] actHeight (stdWidth3 * 0.5) stdHeight]);
        set[hEditNumEV, 'Position', [(2 * border + stdWidth3 * 1.5] actHeight (stdWidth3 * 0.5) stdHeight]);  
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckAgeNormalize, 'Position', [border actHeight stdWidth3 stdHeight]];
        set[hTextAgeSamples, 'Position', [(2 * border + stdWidth3] actHeight (stdWidth3 * 0.5) stdHeight]);
        set[hEditAgeSamples, 'Position', [(2 * border + stdWidth3 * 1.5] actHeight (stdWidth3 * 0.5) stdHeight]);
        set[hTextAgeNorm, 'Position', [(3 * border + stdWidth3 * 2] actHeight (stdWidth3 * 0.5) stdHeight]);
        set[hEditAgeNorm, 'Position', [(3 * border + stdWidth3 * 2.5] actHeight (stdWidth3 * 0.5) stdHeight]);
        
        # Draw feature type selection
        actHeight = actHeight - stdHeight - 2 * border;
        set[hTextFeatureTypes, 'Position', [border actHeight stdWidth stdHeight]];
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckFeatureCompleteStd, 'Position', [border actHeight stdWidth2 stdHeight]];
        set[hCheckFeatureMeanSections, 'Position', [(2 * border + stdWidth2] actHeight stdWidth2 stdHeight]);
        
        actHeight = actHeight - stdHeight - border;
        set[hCheckFeaturePCAAll, 'Position', [border actHeight stdWidth2 stdHeight]];
        set[hCheckFeaturePCANormal, 'Position', [(2 * border + stdWidth2] actHeight stdWidth2 stdHeight]);
        
        # Draw buttons on the bottom of the window
        actHeight = border;
        
        set[hExport, 'Position', [(2 * border + stdWidth2] actHeight stdWidth2 stdHeight]);
        set[hCancel, 'Position', [border actHeight stdWidth2 stdHeight]];         



function hExportCallback[hObject, eventdata]
    [filenameOut, pathnameOut] = uiputfile({'*.txt','Feature Text File';...
        '*.*','All Files' },'Save Features',...
        'features.txt');
   
    if isequal[filenameOut,0]:
        disp['Feature export: Chancelled.T];
        return;
    else:
        disp[['Feature export to ' pathnameOut filenameOut]];
    
    
    [featureCollection, description] = generateFeaturesBScans[DataDescriptors, Layers, Additional, Features, Types];
    
    writeFeatureFile[[pathnameOut filenameOut], featureCollection, description];

    disp['Features succesfully exported.T];


