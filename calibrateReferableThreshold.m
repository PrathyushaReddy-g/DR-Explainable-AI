function calibrateReferableThreshold
% =========================================================================
% REFERABLE DR THRESHOLD CALIBRATION
% Explainable AI for Diabetic Retinopathy Screening
%
% Validation set:
%   data/valid.csv
%
% Test set:
%   data/test.csv
%
% The threshold is learned ONLY from the validation set.
% The final threshold is then evaluated on the test set.
%
% Referable DR:
%   Grade 0 = No DR
%   Grade 1 = Mild DR
%   Grade 2 = Moderate DR
%   Grade 3 = Severe DR
%   Grade 4 = Proliferative DR
%
% Referable = Grade >= 2
% =========================================================================

clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       REFERABLE DR THRESHOLD CALIBRATION\n');
fprintf('============================================================\n\n');

%% ========================================================================
% 1. PATHS
% =========================================================================

projectRoot = fileparts(mfilename('fullpath'));

dataFolder = fullfile(projectRoot,'data');

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'DR_Model.mat');

validCSV = fullfile( ...
    dataFolder, ...
    'valid.csv');

testCSV = fullfile( ...
    dataFolder, ...
    'test.csv');

resultsDir = fullfile( ...
    projectRoot, ...
    'results');

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

%% ========================================================================
% 2. CHECK FILES
% =========================================================================

if ~isfile(modelFile)
    error('Model not found:\n%s',modelFile);
end

if ~isfile(validCSV)
    error('valid.csv not found:\n%s',validCSV);
end

if ~isfile(testCSV)
    error('test.csv not found:\n%s',testCSV);
end

fprintf('Model : %s\n',modelFile);
fprintf('Valid : %s\n',validCSV);
fprintf('Test  : %s\n\n',testCSV);

%% ========================================================================
% 3. LOAD MODEL
% =========================================================================

fprintf('Loading ResNet-18 model...\n');

D = load(modelFile);

if ~isfield(D,'trainedNet')
    error('trainedNet was not found inside DR_Model.mat.');
end

trainedNet = D.trainedNet;

fprintf('Model loaded successfully.\n');

%% ========================================================================
% 4. READ VALIDATION CSV
% =========================================================================

fprintf('\nReading validation CSV...\n');

validTable = readtable(validCSV);

if ~ismember("id_code",string(validTable.Properties.VariableNames))
    error('id_code column not found in valid.csv.');
end

if ~ismember("diagnosis",string(validTable.Properties.VariableNames))
    error('diagnosis column not found in valid.csv.');
end

validIDs = string(validTable.id_code);
validIDs = strtrim(validIDs);

validIDs = regexprep( ...
    validIDs, ...
    '\.(jpg|jpeg|png|tif|tiff)$', ...
    '', ...
    'ignorecase');

validDiagnosis = validTable.diagnosis;

if iscell(validDiagnosis)
    validDiagnosis = string(validDiagnosis);
end

if iscategorical(validDiagnosis)
    validDiagnosis = string(validDiagnosis);
end

validDiagnosis = double(validDiagnosis);

if any(isnan(validDiagnosis))
    error('Invalid diagnosis values found in valid.csv.');
end

if any(~ismember(validDiagnosis,0:4))
    error('Diagnosis values must be between 0 and 4.');
end

fprintf('Validation CSV rows: %d\n',height(validTable));

%% ========================================================================
% 5. FIND VALIDATION IMAGES
% =========================================================================

fprintf('\nSearching for validation images...\n');

extensions = { ...
    '*.jpg', ...
    '*.jpeg', ...
    '*.png', ...
    '*.JPG', ...
    '*.JPEG', ...
    '*.PNG', ...
    '*.tif', ...
    '*.tiff', ...
    '*.TIF', ...
    '*.TIFF'};

allImageFiles = {};

for e = 1:numel(extensions)

    files = dir(fullfile( ...
        dataFolder, ...
        '**', ...
        extensions{e}));

    for k = 1:numel(files)

        if ~files(k).isdir

            allImageFiles{end+1,1} = ...
                fullfile(files(k).folder,files(k).name);

        end

    end
end

allImageFiles = unique(allImageFiles);

fprintf('Total images found under data folder: %d\n', ...
    numel(allImageFiles));

%% ========================================================================
% 6. MATCH VALIDATION IMAGES WITH valid.csv
% =========================================================================

fprintf('\nMatching validation images with valid.csv...\n');

validImageFiles = {};
validLabels = [];

for i = 1:numel(allImageFiles)

    [~,imageID,~] = fileparts(allImageFiles{i});

    matchIndex = find( ...
        strcmpi(validIDs,string(imageID)), ...
        1);

    if ~isempty(matchIndex)

        validImageFiles{end+1,1} = allImageFiles{i};

        validLabels(end+1,1) = ...
            validDiagnosis(matchIndex);

    end
end

fprintf('Validation images matched: %d\n', ...
    numel(validImageFiles));

if isempty(validImageFiles)

    error( ...
        ['No validation images could be matched with valid.csv. ' ...
         'Check the validation image folder.']);

end

%% ========================================================================
% 7. CREATE VALIDATION DATASTORE
% =========================================================================

imdsValid = imageDatastore(validImageFiles);

inputSize = trainedNet.Layers(1).InputSize;

augValid = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsValid, ...
    'ColorPreprocessing','gray2rgb');

%% ========================================================================
% 8. PREDICT VALIDATION SET
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             VALIDATION SET PREDICTION\n');
fprintf('============================================================\n');

fprintf('Processing %d validation images...\n', ...
    numel(validImageFiles));

validScores = minibatchpredict( ...
    trainedNet, ...
    augValid);

if isa(validScores,'dlarray')
    validScores = extractdata(validScores);
end

validScores = gather(validScores);
validScores = double(validScores);

if ndims(validScores) > 2
    validScores = squeeze(validScores);
end

if size(validScores,1) ~= numel(validImageFiles) && ...
        size(validScores,2) == numel(validImageFiles)

    validScores = validScores';
end

if size(validScores,1) ~= numel(validImageFiles)
    error('Validation prediction count does not match image count.');
end

if size(validScores,2) ~= 5
    error('Expected 5 model outputs, but received %d.', ...
        size(validScores,2));
end

%% ========================================================================
% 9. CONVERT TO PROBABILITIES
% =========================================================================

validProbabilities = convertToProbabilities(validScores);

%% ========================================================================
% 10. CALCULATE REFERABLE DR SCORE
% =========================================================================
%
% Probability of Referable DR =
% P(Moderate) + P(Severe) + P(Proliferative)
%
% Classes:
%   column 1 = No DR
%   column 2 = Mild DR
%   column 3 = Moderate DR
%   column 4 = Severe DR
%   column 5 = Proliferative DR
%
% =========================================================================

validReferableScore = sum( ...
    validProbabilities(:,3:5), ...
    2);

trueValidReferable = validLabels >= 2;

%% ========================================================================
% 11. SEARCH THRESHOLDS
% =========================================================================

fprintf('\n');
fprintf('Searching for optimal Referable DR threshold...\n');

thresholds = 0.01:0.01:0.99;

numThresholds = numel(thresholds);

thresholdSensitivity = zeros(numThresholds,1);
thresholdSpecificity = zeros(numThresholds,1);
thresholdPPV = zeros(numThresholds,1);
thresholdNPV = zeros(numThresholds,1);
thresholdF1 = zeros(numThresholds,1);
thresholdAccuracy = zeros(numThresholds,1);

for t = 1:numThresholds

    threshold = thresholds(t);

    predictedReferable = ...
        validReferableScore >= threshold;

    TP = sum(trueValidReferable & predictedReferable);

    FN = sum(trueValidReferable & ~predictedReferable);

    TN = sum(~trueValidReferable & ~predictedReferable);

    FP = sum(~trueValidReferable & predictedReferable);

    thresholdSensitivity(t) = ...
        safeDivide(TP,TP+FN);

    thresholdSpecificity(t) = ...
        safeDivide(TN,TN+FP);

    thresholdPPV(t) = ...
        safeDivide(TP,TP+FP);

    thresholdNPV(t) = ...
        safeDivide(TN,TN+FN);

    thresholdF1(t) = ...
        safeDivide( ...
        2*thresholdPPV(t)*thresholdSensitivity(t), ...
        thresholdPPV(t)+thresholdSensitivity(t));

    thresholdAccuracy(t) = ...
        safeDivide(TP+TN,TP+TN+FP+FN);

end

%% ========================================================================
% 12. SELECT THRESHOLD
% =========================================================================
%
% Primary objective:
%   Maximize sensitivity while maintaining specificity >= 85%.
%
% This is appropriate for screening because missing Referable DR
% is more concerning than sending an additional case for review.
% =========================================================================

specificityRequirement = 0.85;

acceptable = ...
    thresholdSpecificity >= specificityRequirement;

if any(acceptable)

    candidateIndices = find(acceptable);

    candidateSensitivity = ...
        thresholdSensitivity(candidateIndices);

    maxSensitivity = max(candidateSensitivity);

    bestCandidates = candidateIndices( ...
        candidateSensitivity == maxSensitivity);

    % If several thresholds have identical sensitivity,
    % choose the one with the highest specificity.
    [~,bestLocal] = max( ...
        thresholdSpecificity(bestCandidates));

    bestIndex = bestCandidates(bestLocal);

else

    % Fallback:
    % choose threshold with highest Youden index
    youden = ...
        thresholdSensitivity + ...
        thresholdSpecificity - 1;

    [~,bestIndex] = max(youden);

end

optimalThreshold = thresholds(bestIndex);

%% ========================================================================
% 13. DISPLAY CALIBRATION RESULT
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             CALIBRATION RESULT\n');
fprintf('============================================================\n');

fprintf('\nSelected threshold: %.2f\n', ...
    optimalThreshold);

fprintf('\nValidation performance at selected threshold:\n');

fprintf('Sensitivity : %.2f%%\n', ...
    100*thresholdSensitivity(bestIndex));

fprintf('Specificity : %.2f%%\n', ...
    100*thresholdSpecificity(bestIndex));

fprintf('PPV         : %.2f%%\n', ...
    100*thresholdPPV(bestIndex));

fprintf('NPV         : %.2f%%\n', ...
    100*thresholdNPV(bestIndex));

fprintf('F1 Score    : %.2f%%\n', ...
    100*thresholdF1(bestIndex));

fprintf('Accuracy    : %.2f%%\n', ...
    100*thresholdAccuracy(bestIndex));

%% ========================================================================
% 14. SAVE THRESHOLD
% =========================================================================

thresholdFile = fullfile( ...
    resultsDir, ...
    'referable_DR_threshold.mat');

save( ...
    thresholdFile, ...
    'optimalThreshold');

fprintf('\nThreshold saved:\n%s\n',thresholdFile);

%% ========================================================================
% 15. SAVE THRESHOLD SEARCH RESULTS
% =========================================================================

thresholdTable = table( ...
    thresholds(:), ...
    100*thresholdSensitivity(:), ...
    100*thresholdSpecificity(:), ...
    100*thresholdPPV(:), ...
    100*thresholdNPV(:), ...
    100*thresholdF1(:), ...
    100*thresholdAccuracy(:), ...
    'VariableNames',{ ...
    'Threshold', ...
    'Sensitivity_Percent', ...
    'Specificity_Percent', ...
    'PPV_Percent', ...
    'NPV_Percent', ...
    'F1_Percent', ...
    'Accuracy_Percent'});

thresholdCSV = fullfile( ...
    resultsDir, ...
    'referable_DR_threshold_search.csv');

writetable( ...
    thresholdTable, ...
    thresholdCSV);

%% ========================================================================
% 16. THRESHOLD PERFORMANCE GRAPH
% =========================================================================

fig = figure( ...
    'Name','Referable DR Threshold Calibration', ...
    'Color','w', ...
    'Position',[150 100 1000 650]);

plot( ...
    thresholds, ...
    100*thresholdSensitivity, ...
    'LineWidth',2);

hold on;

plot( ...
    thresholds, ...
    100*thresholdSpecificity, ...
    'LineWidth',2);

xline( ...
    optimalThreshold, ...
    '--', ...
    'LineWidth',2);

yline( ...
    90, ...
    ':', ...
    'LineWidth',1.5);

yline( ...
    85, ...
    ':', ...
    'LineWidth',1.5);

xlabel('Referable DR Probability Threshold');

ylabel('Percentage (%)');

title('Referable DR Threshold Calibration');

legend( ...
    'Sensitivity', ...
    'Specificity', ...
    'Selected Threshold', ...
    '90% Sensitivity Target', ...
    '85% Specificity Target', ...
    'Location','best');

grid on;

thresholdFigure = fullfile( ...
    resultsDir, ...
    'referable_DR_threshold_calibration.png');

exportgraphics( ...
    fig, ...
    thresholdFigure, ...
    'Resolution',300);

%% ========================================================================
% 17. APPLY FIXED THRESHOLD TO TEST SET
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              FINAL TEST SET EVALUATION\n');
fprintf('============================================================\n');

testTable = readtable(testCSV);

testIDs = string(testTable.id_code);
testIDs = strtrim(testIDs);

testIDs = regexprep( ...
    testIDs, ...
    '\.(jpg|jpeg|png|tif|tiff)$', ...
    '', ...
    'ignorecase');

testDiagnosis = testTable.diagnosis;

if iscell(testDiagnosis)
    testDiagnosis = string(testDiagnosis);
end

if iscategorical(testDiagnosis)
    testDiagnosis = string(testDiagnosis);
end

testDiagnosis = double(testDiagnosis);

%% Find test images

testImageFiles = {};

for i = 1:numel(allImageFiles)

    [~,imageID,~] = fileparts(allImageFiles{i});

    matchIndex = find( ...
        strcmpi(testIDs,string(imageID)), ...
        1);

    if ~isempty(matchIndex)

        testImageFiles{end+1,1} = ...
            allImageFiles{i};

    end
end

fprintf('Test images matched: %d\n', ...
    numel(testImageFiles));

if isempty(testImageFiles)
    error('No test images could be matched.');
end

%% Match labels in same order

testLabels = zeros(numel(testImageFiles),1);

for i = 1:numel(testImageFiles)

    [~,imageID,~] = fileparts(testImageFiles{i});

    matchIndex = find( ...
        strcmpi(testIDs,string(imageID)), ...
        1);

    testLabels(i) = testDiagnosis(matchIndex);

end

%% Test datastore

imdsTest = imageDatastore(testImageFiles);

augTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing','gray2rgb');

fprintf('Running final test predictions...\n');

testScores = minibatchpredict( ...
    trainedNet, ...
    augTest);

if isa(testScores,'dlarray')
    testScores = extractdata(testScores);
end

testScores = gather(testScores);
testScores = double(testScores);

if ndims(testScores) > 2
    testScores = squeeze(testScores);
end

if size(testScores,1) ~= numel(testImageFiles) && ...
        size(testScores,2) == numel(testImageFiles)

    testScores = testScores';
end

if size(testScores,1) ~= numel(testImageFiles)
    error('Test prediction count does not match image count.');
end

testProbabilities = ...
    convertToProbabilities(testScores);

%% ========================================================================
% 18. APPLY CALIBRATED THRESHOLD
% =========================================================================

testReferableScore = sum( ...
    testProbabilities(:,3:5), ...
    2);

trueTestReferable = ...
    testLabels >= 2;

predTestReferable = ...
    testReferableScore >= optimalThreshold;

%% ========================================================================
% 19. FINAL TEST METRICS
% =========================================================================

TP = sum(trueTestReferable & predTestReferable);

FN = sum(trueTestReferable & ~predTestReferable);

TN = sum(~trueTestReferable & ~predTestReferable);

FP = sum(~trueTestReferable & predTestReferable);

testSensitivity = ...
    safeDivide(TP,TP+FN);

testSpecificity = ...
    safeDivide(TN,TN+FP);

testPPV = ...
    safeDivide(TP,TP+FP);

testNPV = ...
    safeDivide(TN,TN+FN);

testF1 = ...
    safeDivide( ...
    2*testPPV*testSensitivity, ...
    testPPV+testSensitivity);

testAccuracy = ...
    safeDivide(TP+TN,TP+TN+FP+FN);

%% ========================================================================
% 20. DISPLAY FINAL TEST RESULT
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('       FINAL TEST RESULT AFTER CALIBRATION\n');
fprintf('============================================================\n');

fprintf('\nFixed threshold: %.2f\n', ...
    optimalThreshold);

fprintf('\n');

fprintf('Sensitivity : %.2f%%\n', ...
    100*testSensitivity);

fprintf('Specificity : %.2f%%\n', ...
    100*testSpecificity);

fprintf('PPV         : %.2f%%\n', ...
    100*testPPV);

fprintf('NPV         : %.2f%%\n', ...
    100*testNPV);

fprintf('F1 Score    : %.2f%%\n', ...
    100*testF1);

fprintf('Accuracy    : %.2f%%\n', ...
    100*testAccuracy);

fprintf('\n');

fprintf('Binary Confusion Matrix:\n');

fprintf('                 Predicted\n');
fprintf('                 Non-Ref   Ref\n');
fprintf('True Non-Ref     %6d   %6d\n',TN,FP);
fprintf('True Ref         %6d   %6d\n',FN,TP);

%% ========================================================================
% 21. SAVE FINAL COMPARISON
% =========================================================================

comparisonName = [
    "Original Test Threshold"
    "Calibrated Test Threshold"
    ];

comparisonSensitivity = [
    80.29
    100*testSensitivity
    ];

comparisonSpecificity = [
    95.63
    100*testSpecificity
    ];

comparisonPPV = [
    91.67
    100*testPPV
    ];

comparisonNPV = [
    89.02
    100*testNPV
    ];

comparisonF1 = [
    85.60
    100*testF1
    ];

comparisonTable = table( ...
    comparisonName, ...
    comparisonSensitivity, ...
    comparisonSpecificity, ...
    comparisonPPV, ...
    comparisonNPV, ...
    comparisonF1, ...
    'VariableNames',{ ...
    'Method', ...
    'Sensitivity_Percent', ...
    'Specificity_Percent', ...
    'PPV_Percent', ...
    'NPV_Percent', ...
    'F1_Percent'});

comparisonFile = fullfile( ...
    resultsDir, ...
    'referable_DR_before_after_calibration.csv');

writetable( ...
    comparisonTable, ...
    comparisonFile);

%% ========================================================================
% 22. SAVE FINAL TEST PREDICTIONS
% =========================================================================

testPredictionTable = table( ...
    string(testImageFiles(:)), ...
    testLabels(:), ...
    testReferableScore(:), ...
    trueTestReferable(:), ...
    predTestReferable(:), ...
    'VariableNames',{ ...
    'Image', ...
    'True_Diagnosis', ...
    'Referable_DR_Probability', ...
    'True_Referable', ...
    'Predicted_Referable'});

testPredictionFile = fullfile( ...
    resultsDir, ...
    'calibrated_test_predictions.csv');

writetable( ...
    testPredictionTable, ...
    testPredictionFile);

%% ========================================================================
% 23. COMPLETE
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('            THRESHOLD CALIBRATION COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nResults saved in:\n%s\n',resultsDir);

fprintf('\nCreated files:\n');
fprintf('1. referable_DR_threshold.mat\n');
fprintf('2. referable_DR_threshold_search.csv\n');
fprintf('3. referable_DR_threshold_calibration.png\n');
fprintf('4. referable_DR_before_after_calibration.csv\n');
fprintf('5. calibrated_test_predictions.csv\n');

fprintf('\n');

end


%% =========================================================================
% FUNCTION: CONVERT SCORES TO PROBABILITIES
% =========================================================================

function probabilities = convertToProbabilities(scores)

scores = double(scores);

rowSums = sum(scores,2);

isProbability = ...
    all(scores(:) >= 0) && ...
    all(scores(:) <= 1) && ...
    all(abs(rowSums - 1) < 0.05);

if isProbability

    probabilities = ...
        scores ./ max(rowSums,eps);

else

    shiftedScores = ...
        scores - max(scores,[],2);

    expScores = exp(shiftedScores);

    probabilities = ...
        expScores ./ max(sum(expScores,2),eps);

end

end


%% =========================================================================
% FUNCTION: SAFE DIVISION
% =========================================================================

function value = safeDivide(numerator,denominator)

if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end

end