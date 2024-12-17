function perform_all_segmentations()
    global ActDataDescriptors guiMode

    % Start the main OCTSEG application
    octsegMain;

    % Perform all segmentations
    [goOn, notProcessedList] = getFileNumbersForAutomatedProcessing(ActDataDescriptors, guiMode, 'ONFL');
    if ~goOn, return; end

    hMenAutoOCTSEGCallback([], notProcessedList);
    hMenAutoRPECallback([], notProcessedList);
    hMenAutoONHCallback([], notProcessedList);
    hMenAutoINFLCallback([], notProcessedList);
    hMenAutoBVCallback([], notProcessedList);
    hMenAutoInnerLayersCallback([], notProcessedList);
    hMenAutoONFLCallback([], notProcessedList);
end