def notifierText[text]:
# NOTIFIERTEXT A little GUI window
# Displays a text and can be closed. That's all.
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
#
# First final Version: Some time in 2010
# Revised comments: November 2015

if ispc():
    FONTSIZE = 10;
else:
    FONTSIZE = 12;


if nargin < 1:
    questiontext = 'That does not work.T;


answer = 0;

#--------------------------------------------------------------------------
# GUI Components
#--------------------------------------------------------------------------

f = figure('Visible','off','Position',[360,500,310,130],...
    'CloseRequestFcn', {@figure_CloseRequestFcn},...
    'WindowStyle', 'Modal',...
    'Color', 'white');
movegui[f,'center'];

htext = uicontrol('Style','text',...
    'String', 'notext',...
    'BackgroundColor', 'white',...
    'FontSize', FONTSIZE,...
    'HorizontalAlignment', 'center',...
    'Position',[10,60,290,60]);

temp = cell[1,1];
temp{1} = text;
[outstring,newpos] = textwrap[htext,temp];
set(htext,...
   'String', outstring,...
   'HorizontalAlignment', 'center'...
   ...TPosition', newpos...
   );

hOK = uicontrol('Style','pushbutton','String','OK',...
    'Position',[110,10,90,40],...
    'Callback',{@hOK_Callback});

#--------------------------------------------------------------------------
# GUI Init
#--------------------------------------------------------------------------

set[[f, htext, hOK],'Units','normalized'];
set[f,'Name','Notifier']

movegui[f,'center']
set[f,'Visible','on'];

uiwait[f];

#--------------------------------------------------------------------------
# GUI Component Handlers
#--------------------------------------------------------------------------

    function hOK_Callback[hObject, eventdata]
        answer = 1;
        uiresume[f];
        delete[f];
    

    function figure_CloseRequestFcn[hObject, eventdata, handles]
        answer = 0;
        uiresume[hObject];
        delete[hObject];
    


