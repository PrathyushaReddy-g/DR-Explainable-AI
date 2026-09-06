function clinicalValidationMetrics
% =========================================================================
% CLINICAL VALIDATION USING LABELED TEST CSV
% Explainable AI for Diabetic Retinopathy Screening
%
% Dataset:
%   data/test.csv
%   data/test_images/test_images/
%
% CSV columns:
%   id_code
%   diagnosis
%
% diagnosis:
%   0 = No DR
%   1 = Mild DR
%   2 = Moderate DR
%   3 = Severe DR
%   4 = Proliferative DR
%
% Referable DR:
%   diagnosis >= 2
% =========================================================================

clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       CLINICAL VALIDATION - DIABETIC RETINOPATHY\n');
fprintf('============================================================\n\n');

%% ========================================================================
% 1. PATHS
% =========================================================================

projectRoot = fileparts(mfilename('fullpath'));

dataFolder = fullfile(projectRoot, 'data');

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'DR_Model.mat');

resultsDir = fullfile( ...
    projectRoot, ...
    'results');

csvFile = fullfile( ...
    dataFolder, ...
    'test.csv');

imageFolder = fullfile( ...
    dataFolder, ...
    'test_images');

%% Create results folder if necessary

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

%% ========================================================================
% 2. CHECK FILES
% =========================================================================

if ~isfolder(dataFolder)
    error('Data folder not found:\n%s', dataFolder);
end

if ~isfile(modelFile)
    error('DR_Model.mat not found:\n%s', modelFile);
end

if ~isfile(csvFile)
    error('test.csv not found:\n%s', csvFile);
end

if ~isfolder(imageFolder)
    error('Test image folder not found:\n%s', imageFolder);
end

fprintf('Project folder : %s\n', projectRoot);
fprintf('CSV file       : %s\n', csvFile);
fprintf('Image folder   : %s\n', imageFolder);
fprintf('Model file     : %s\n\n', modelFile);

%% ========================================================================
% 3. LOAD MODEL
% =========================================================================

fprintf('Loading trained ResNet-18 model...\n');

D = load(modelFile);

if ~isfield(D, 'trainedNet')
    error('Variable "trainedNet" was not found inside DR_Model.mat.');
end

trainedNet = D.trainedNet;

%% Class names

if isfield(D, 'classNames')
    classNames = string(D.classNames);
else
    classNames = [
        "No DR"
        "Mild DR"
        "Moderate DR"
        "Severe DR"
        "Proliferative DR"
        ];
end

if numel(classNames) < 5
    error('The model must contain 5 diabetic retinopathy classes.');
end

classNames = classNames(1:5);

fprintf('Model loaded successfully.\n');

%% ========================================================================
% 4. READ TEST CSV
% =========================================================================

fprintf('\nReading test.csv...\n');

testTable = readtable(csvFile);

fprintf('CSV rows found: %d\n', height(testTable));

%% Check required columns

if ~ismember("id_code", string(testTable.Properties.VariableNames))
    error('Column "id_code" was not found in test.csv.');
end

if ~ismember("diagnosis", string(testTable.Properties.VariableNames))
    error('Column "diagnosis" was not found in test.csv.');
end

%% ========================================================================
% 5. CLEAN CSV IDS
% =========================================================================

csvIDs = string(testTable.id_code);

csvIDs = strtrim(csvIDs);

% Remove accidental file extensions if present
csvIDs = regexprep(csvIDs, '\.(jpg|jpeg|png|tif|tiff)$', '', ...
    'ignorecase');

%% Convert diagnosis to numeric

diagnosisValues = testTable.diagnosis;

if iscell(diagnosisValues)
    diagnosisValues = string(diagnosisValues);
end

if iscategorical(diagnosisValues)
    diagnosisValues = string(diagnosisValues);
end

diagnosisValues = double(diagnosisValues);

%% Validate diagnosis values

if any(isnan(diagnosisValues))
    error('Some diagnosis values in test.csv are missing or invalid.');
end

if any(~ismember(diagnosisValues, 0:4))
    error('Diagnosis values must be integers from 0 to 4.');
end

fprintf('Valid labeled images in CSV: %d\n', height(testTable));

%% ========================================================================
% 6. FIND TEST IMAGES
% =========================================================================

fprintf('\nSearching for test images...\n');

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

imageFiles = {};

for e = 1:numel(extensions)

    tempFiles = dir(fullfile( ...
        imageFolder, ...
        '**', ...
        extensions{e}));

    for k = 1:numel(tempFiles)

        if ~tempFiles(k).isdir

            imageFiles{end+1,1} = fullfile( ...
                tempFiles(k).folder, ...
                tempFiles(k).name);

        end

    end
end

%% Remove duplicates

if ~isempty(imageFiles)
    imageFiles = unique(imageFiles);
end

fprintf('Images found: %d\n', numel(imageFiles));

if isempty(imageFiles)
    error('No test images were found.');
end

%% ========================================================================
% 7. MATCH IMAGE FILENAMES WITH CSV id_code
% =========================================================================

fprintf('\nMatching images with CSV labels...\n');

numImages = numel(imageFiles);

matchedFiles = {};
matchedDiagnosis = [];

missingIDs = strings(0,1);

for i = 1:numImages

    [~, imageName, ~] = fileparts(imageFiles{i});

    imageID = string(imageName);

    % Remove extension already handled by fileparts
    imageID = strtrim(imageID);

    matchIndex = find( ...
        strcmpi(csvIDs, imageID), ...
        1);

    if ~isempty(matchIndex)

        matchedFiles{end+1,1} = imageFiles{i};

        matchedDiagnosis(end+1,1) = ...
            diagnosisValues(matchIndex);

    else

        missingIDs(end+1,1) = imageID;

    end
end

fprintf('Images successfully matched: %d\n', ...
    numel(matchedFiles));

fprintf('Images without CSV match: %d\n', ...
    numel(missingIDs));

%% Stop if no images matched

if isempty(matchedFiles)

    fprintf('\nExample image IDs:\n');

    for i = 1:min(5,numImages)
        [~, name, ~] = fileparts(imageFiles{i});
        fprintf('%s\n', name);
    end

    fprintf('\nExample CSV IDs:\n');

    for i = 1:min(5,height(testTable))
        fprintf('%s\n', csvIDs(i));
    end

    error('No images could be matched with test.csv.');
end

%% ========================================================================
% 8. CREATE IMAGE DATASTORE
% =========================================================================

imds = imageDatastore(matchedFiles);

fprintf('\nImage datastore created.\n');
fprintf('Validation images: %d\n', numel(matchedFiles));

%% ========================================================================
% 9. CLASS DISTRIBUTION
% =========================================================================

fprintf('\n============================================================\n');
fprintf('                     CLASS DISTRIBUTION\n');
fprintf('============================================================\n');

for i = 0:4

    count = sum(matchedDiagnosis == i);

    fprintf( ...
        'Grade %d - %-20s : %d images\n', ...
        i, ...
        classNames(i+1), ...
        count);

end

%% ========================================================================
% 10. MODEL INPUT SIZE
% =========================================================================

inputSize = trainedNet.Layers(1).InputSize;

fprintf('\nNetwork input size: %d x %d x %d\n', ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3));

%% ========================================================================
% 11. CREATE AUGMENTED DATASTORE
% =========================================================================

augTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imds, ...
    'ColorPreprocessing', 'gray2rgb');

%% ========================================================================
% 12. RUN MODEL PREDICTIONS
% =========================================================================

fprintf('\n============================================================\n');
fprintf('                 RUNNING RESNET-18\n');
fprintf('============================================================\n');

fprintf('Processing %d test images...\n', ...
    numel(matchedFiles));

scores = minibatchpredict( ...
    trainedNet, ...
    augTest);

%% Convert dlarray if needed

if isa(scores, 'dlarray')
    scores = extractdata(scores);
end

scores = gather(scores);
scores = double(scores);

%% ========================================================================
% 13. FIX SCORE ORIENTATION
% =========================================================================

if ndims(scores) > 2
    scores = squeeze(scores);
end

if size(scores,1) ~= numel(matchedFiles) && ...
        size(scores,2) == numel(matchedFiles)

    scores = scores';
end

if size(scores,1) ~= numel(matchedFiles)
    error( ...
        ['Prediction count does not match image count.\n' ...
         'Images: %d\nPredictions: %d'], ...
        numel(matchedFiles), ...
        size(scores,1));
end

if size(scores,2) ~= 5
    error( ...
        'The model returned %d outputs. Expected 5 classes.', ...
        size(scores,2));
end

%% ========================================================================
% 14. CONVERT SCORES TO PROBABILITIES
% =========================================================================

probabilities = convertToProbabilities(scores);

%% ========================================================================
% 15. PREDICTED CLASS
% =========================================================================

[confidence, predIndex] = ...
    max(probabilities, [], 2);

%% predIndex:
% 1 = No DR
% 2 = Mild
% 3 = Moderate
% 4 = Severe
% 5 = Proliferative

predDiagnosis = predIndex - 1;

trueDiagnosis = matchedDiagnosis;

%% ========================================================================
% 16. FIVE-CLASS CONFUSION MATRIX
% =========================================================================

fprintf('\nCalculating confusion matrix...\n');

cm = zeros(5,5);

for i = 1:numel(trueDiagnosis)

    trueClass = trueDiagnosis(i) + 1;
    predClass = predDiagnosis(i) + 1;

    cm(trueClass,predClass) = ...
        cm(trueClass,predClass) + 1;

end

N = sum(cm(:));

%% ========================================================================
% 17. OVERALL ACCURACY
% =========================================================================

accuracy = trace(cm) / max(N,1);

%% ========================================================================
% 18. PER-CLASS METRICS
% =========================================================================

numClasses = 5;

sensitivity = zeros(numClasses,1);
specificity = zeros(numClasses,1);
precision   = zeros(numClasses,1);
npv         = zeros(numClasses,1);
f1          = zeros(numClasses,1);

for i = 1:numClasses

    TP = cm(i,i);

    FN = sum(cm(i,:)) - TP;

    FP = sum(cm(:,i)) - TP;

    TN = N - TP - FN - FP;

    sensitivity(i) = ...
        safeDivide(TP, TP + FN);

    specificity(i) = ...
        safeDivide(TN, TN + FP);

    precision(i) = ...
        safeDivide(TP, TP + FP);

    npv(i) = ...
        safeDivide(TN, TN + FN);

    f1(i) = ...
        safeDivide( ...
        2 * precision(i) * sensitivity(i), ...
        precision(i) + sensitivity(i));

end

%% ========================================================================
% 19. REFERABLE DR METRICS
% =========================================================================
%
% Referable DR:
%   Grade 2 = Moderate
%   Grade 3 = Severe
%   Grade 4 = Proliferative
%
% Therefore:
%   Referable = diagnosis >= 2
% =========================================================================

trueReferable = trueDiagnosis >= 2;

predReferable = predDiagnosis >= 2;

%% Binary confusion matrix

TPref = sum(trueReferable & predReferable);

FNref = sum(trueReferable & ~predReferable);

TNref = sum(~trueReferable & ~predReferable);

FPref = sum(~trueReferable & predReferable);

%% Referable metrics

referableSensitivity = ...
    safeDivide(TPref, TPref + FNref);

referableSpecificity = ...
    safeDivide(TNref, TNref + FPref);

referablePPV = ...
    safeDivide(TPref, TPref + FPref);

referableNPV = ...
    safeDivide(TNref, TNref + FNref);

referableF1 = ...
    safeDivide( ...
    2 * referablePPV * referableSensitivity, ...
    referablePPV + referableSensitivity);

%% ========================================================================
% 20. AGREEMENT
% =========================================================================

gradeAgreement = ...
    mean(trueDiagnosis == predDiagnosis);

referralAgreement = ...
    mean(trueReferable == predReferable);

%% ========================================================================
% 21. DISPLAY MAIN RESULTS
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              CLINICAL VALIDATION RESULTS\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf('Test Images                    : %d\n', N);

fprintf('Overall Accuracy               : %.2f%%\n', ...
    100 * accuracy);

fprintf('Grade Agreement                : %.2f%%\n', ...
    100 * gradeAgreement);

fprintf('Referral Agreement             : %.2f%%\n', ...
    100 * referralAgreement);

fprintf('\n');

fprintf('------------------------------------------------------------\n');
fprintf('                  REFERABLE DR (GRADE 2+)\n');
fprintf('------------------------------------------------------------\n');

fprintf('Sensitivity                    : %.2f%%\n', ...
    100 * referableSensitivity);

fprintf('Specificity                    : %.2f%%\n', ...
    100 * referableSpecificity);

fprintf('PPV                            : %.2f%%\n', ...
    100 * referablePPV);

fprintf('NPV                            : %.2f%%\n', ...
    100 * referableNPV);

fprintf('F1 Score                       : %.2f%%\n', ...
    100 * referableF1);

fprintf('\n');

%% ========================================================================
% 22. PER-CLASS RESULTS
% =========================================================================

fprintf('------------------------------------------------------------\n');
fprintf('                     PER-CLASS RESULTS\n');
fprintf('------------------------------------------------------------\n');

for i = 1:numClasses

    fprintf( ...
        '%-20s Sensitivity: %6.2f%%   Specificity: %6.2f%%   F1: %6.2f%%\n', ...
        classNames(i), ...
        100 * sensitivity(i), ...
        100 * specificity(i), ...
        100 * f1(i));

end

%% ========================================================================
% 23. DISPLAY CONFUSION MATRIX NUMERICALLY
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  5-CLASS CONFUSION MATRIX\n');
fprintf('============================================================\n');

disp(cm);

%% ========================================================================
% 24. REFERABLE TARGET CHECK
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    SIH TARGET CHECK\n');
fprintf('============================================================\n');

fprintf('Target Sensitivity : >90%%\n');
fprintf('Actual Sensitivity : %.2f%%\n', ...
    100 * referableSensitivity);

fprintf('\n');

fprintf('Target Specificity : >85%%\n');
fprintf('Actual Specificity : %.2f%%\n', ...
    100 * referableSpecificity);

fprintf('\n');

if referableSensitivity >= 0.90 && ...
        referableSpecificity >= 0.85

    fprintf('STATUS: TARGET ACHIEVED\n');

else

    fprintf('STATUS: TARGET NOT YET ACHIEVED\n');

end

%% ========================================================================
% 25. CONFUSION MATRIX FIGURE
% =========================================================================

fprintf('\nGenerating confusion matrix figure...\n');

trueCategorical = categorical( ...
    trueDiagnosis, ...
    0:4, ...
    cellstr(classNames));

predCategorical = categorical( ...
    predDiagnosis, ...
    0:4, ...
    cellstr(classNames));

fig = figure( ...
    'Name', 'Clinical Validation - Confusion Matrix', ...
    'Color', 'w', ...
    'Position', [150 100 950 750]);

confusionchart( ...
    trueCategorical, ...
    predCategorical, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

title('ResNet-18 Diabetic Retinopathy Validation');

cmFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_confusion_matrix.png');

exportgraphics( ...
    fig, ...
    cmFile, ...
    'Resolution', 300);

%% ========================================================================
% 26. SAVE METRICS CSV
% =========================================================================

metricName = [
    "Overall Accuracy"
    "Grade Agreement"
    "Referral Agreement"
    "Referable Sensitivity"
    "Referable Specificity"
    "Referable PPV"
    "Referable NPV"
    "Referable F1"
    ];

metricValue = [
    accuracy
    gradeAgreement
    referralAgreement
    referableSensitivity
    referableSpecificity
    referablePPV
    referableNPV
    referableF1
    ];

metricsTable = table( ...
    metricName, ...
    100 * metricValue, ...
    'VariableNames', ...
    {'Metric','Percentage'});

metricsFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_metrics.csv');

writetable( ...
    metricsTable, ...
    metricsFile);

%% ========================================================================
% 27. SAVE PER-CLASS METRICS
% =========================================================================

classTable = table( ...
    classNames(:), ...
    100 * sensitivity, ...
    100 * specificity, ...
    100 * precision, ...
    100 * npv, ...
    100 * f1, ...
    'VariableNames', { ...
    'Class', ...
    'Sensitivity_Percent', ...
    'Specificity_Percent', ...
    'PPV_Percent', ...
    'NPV_Percent', ...
    'F1_Percent'});

classFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_per_class.csv');

writetable( ...
    classTable, ...
    classFile);

%% ========================================================================
% 28. SAVE IMAGE-LEVEL PREDICTIONS
% =========================================================================

fprintf('\nSaving image-level prediction results...\n');

% Force every variable to be a column vector
imageColumn = reshape(string(matchedFiles), [], 1);

trueDiagnosisColumn = reshape(trueDiagnosis, [], 1);

predDiagnosisColumn = reshape(predDiagnosis, [], 1);

confidenceColumn = reshape(100 * confidence, [], 1);

trueReferableColumn = reshape(trueReferable, [], 1);

predReferableColumn = reshape(predReferable, [], 1);

% Convert numerical grades to readable class names
trueGradeColumn = reshape( ...
    classNames(trueDiagnosisColumn + 1), [], 1);

predGradeColumn = reshape( ...
    classNames(predDiagnosisColumn + 1), [], 1);

% Make sure all columns have exactly the same number of rows
numPredictionRows = numel(imageColumn);

if numel(trueDiagnosisColumn) ~= numPredictionRows || ...
        numel(predDiagnosisColumn) ~= numPredictionRows || ...
        numel(trueGradeColumn) ~= numPredictionRows || ...
        numel(predGradeColumn) ~= numPredictionRows || ...
        numel(confidenceColumn) ~= numPredictionRows || ...
        numel(trueReferableColumn) ~= numPredictionRows || ...
        numel(predReferableColumn) ~= numPredictionRows

    error('Image-level prediction variables have different lengths.');
end

% Create table
imageTable = table( ...
    imageColumn, ...
    trueDiagnosisColumn, ...
    predDiagnosisColumn, ...
    trueGradeColumn, ...
    predGradeColumn, ...
    confidenceColumn, ...
    trueReferableColumn, ...
    predReferableColumn, ...
    'VariableNames', { ...
    'Image', ...
    'True_Diagnosis', ...
    'Predicted_Diagnosis', ...
    'True_Grade', ...
    'Predicted_Grade', ...
    'Confidence_Percent', ...
    'True_Referable', ...
    'Predicted_Referable'});

% Save CSV
predictionFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_predictions.csv');

writetable( ...
    imageTable, ...
    predictionFile);

fprintf('Image-level predictions saved successfully.\n');
%% ========================================================================
% 29. SAVE 5-CLASS CONFUSION MATRIX
% =========================================================================

cmTable = array2table( ...
    cm, ...
    'VariableNames', cellstr(classNames), ...
    'RowNames', cellstr(classNames));

cmCsvFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_confusion_matrix.csv');

writetable( ...
    cmTable, ...
    cmCsvFile, ...
    'WriteRowNames', true);

%% ========================================================================
% 30. SAVE REFERABLE CONFUSION MATRIX
% =========================================================================

refTable = table( ...
    TPref, ...
    FNref, ...
    TNref, ...
    FPref, ...
    'VariableNames', { ...
    'True_Positive', ...
    'False_Negative', ...
    'True_Negative', ...
    'False_Positive'});

refFile = fullfile( ...
    resultsDir, ...
    'referable_DR_confusion_matrix.csv');

writetable( ...
    refTable, ...
    refFile);

%% ========================================================================
% 31. SAVE SUMMARY TEXT
% =========================================================================

summaryFile = fullfile( ...
    resultsDir, ...
    'clinical_validation_summary.txt');

fid = fopen(summaryFile, 'w');

if fid ~= -1

    fprintf(fid, 'DIABETIC RETINOPATHY CLINICAL VALIDATION\n');
    fprintf(fid, '========================================\n\n');

    fprintf(fid, 'Model: ResNet-18\n');
    fprintf(fid, 'Test images: %d\n\n', N);

    fprintf(fid, 'Overall Accuracy: %.2f%%\n', ...
        100 * accuracy);

    fprintf(fid, 'Grade Agreement: %.2f%%\n', ...
        100 * gradeAgreement);

    fprintf(fid, 'Referral Agreement: %.2f%%\n\n', ...
        100 * referralAgreement);

    fprintf(fid, 'REFERABLE DR (GRADE 2+)\n');
    fprintf(fid, 'Sensitivity: %.2f%%\n', ...
        100 * referableSensitivity);

    fprintf(fid, 'Specificity: %.2f%%\n', ...
        100 * referableSpecificity);

    fprintf(fid, 'PPV: %.2f%%\n', ...
        100 * referablePPV);

    fprintf(fid, 'NPV: %.2f%%\n', ...
        100 * referableNPV);

    fprintf(fid, 'F1: %.2f%%\n', ...
        100 * referableF1);

    fclose(fid);

end

%% ========================================================================
% 32. COMPLETE
% =========================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 VALIDATION COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nResults saved to:\n');
fprintf('%s\n', resultsDir);

fprintf('\nFiles created:\n');
fprintf('1. clinical_validation_metrics.csv\n');
fprintf('2. clinical_validation_per_class.csv\n');
fprintf('3. clinical_validation_predictions.csv\n');
fprintf('4. referable_DR_confusion_matrix.csv\n');
fprintf('5. clinical_validation_confusion_matrix.csv\n');
fprintf('6. clinical_validation_confusion_matrix.png\n');
fprintf('7. clinical_validation_summary.txt\n');

fprintf('\n============================================================\n\n');

end


%% =========================================================================
% FUNCTION: CONVERT SCORES TO PROBABILITIES
% =========================================================================

function probabilities = convertToProbabilities(scores)

scores = double(scores);

%% Check whether scores already look like probabilities

rowSums = sum(scores, 2);

isProbability = ...
    all(scores(:) >= 0) && ...
    all(scores(:) <= 1) && ...
    all(abs(rowSums - 1) < 0.05);

if isProbability

    probabilities = ...
        scores ./ max(rowSums, eps);

else

    %% Softmax

    shiftedScores = ...
        scores - max(scores, [], 2);

    expScores = exp(shiftedScores);

    probabilities = ...
        expScores ./ max(sum(expScores, 2), eps);

end

end


%% =========================================================================
% FUNCTION: SAFE DIVISION
% =========================================================================

function value = safeDivide(numerator, denominator)

if denominator == 0
    value = 0;
else
    value = numerator / denominator;
end

end