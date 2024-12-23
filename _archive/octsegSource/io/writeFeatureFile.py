function writeFeatureFile[filename, featureCollection, description]

fido = fopen[filename, 'w'];

numAdditional = numel[featureCollection{1,1}];
numFeatures = numel[featureCollection{1,3}];

fprintf[fido, '#.0f %.0f\n', numAdditional, numFeatures];
for i in 1:numel[description]:
    fprintf[fido, '#s\t', description{i}];

fprintf[fido, '\n'];

for i in 1:size[featureCollection, 1]:
    for k in 1:numAdditional:
        if numel[featureCollection{i, 1}{k}] > 1:
            fprintf[fido, '#s\t', featureCollection{i, 1}{k}];
        else:
            fprintf[fido, '#f\t', featureCollection{i, 1}{k}];
        
    
    
    fprintf[fido, '#.0f\t', featureCollection{i, 2}];
    
    for k in 1:numel[featureCollection{i, 3}]:
        fprintf[fido, '#.5f\t', featureCollection{i, 3}(k]);
    
    
    fprintf[fido, '\n'];


fclose[fido];
