function validate_DR_Model
%% ================================================================
%  VALIDATE_DR_MODEL
%
%  Independent test-set validation for the existing
%  Explainable AI Diabetic Retinopathy classifier.
%
%  IMPORTANT:
%  - DOES NOT retrain the model.
%  - Uses the existing DR_Model.mat.
%  - Uses the actual test.csv.
%  - Uses imageDatastore + augmentedImageDatastore.
%  - Uses minibatchpredict, matching the MATLAB training workflow.
%
%  Evaluates:
%    1. Overall accuracy
%    2. Confusion matrix
%    3. Predicted-class distribution
%    4. Per-class precision
%    5. Per-class sensitivity
%    6. Per-class specificity
%    7. Per-class F1
%    8. Macro averages
%    9. Referable DR performance (Grade >= 2)
%   10. Target comparison
%   11. Saves complete results
%
%  DR classes:
%    0 = No DR
%    1 = Mild DR
%    2 = Moderate DR
%    3 = Severe DR
%    4 = Proliferative DR
%
%  Referable DR = Grade >= 2
%
% ================================================================

clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       DIABETIC RETINOPATHY TEST SET VALIDATION\n');
fprintf('============================================================\n');
fprintf('\n');

%% ================================================================
% 1. PROJECT SETUP
% ================================================================

projectRoot = fileparts(mfilename('fullpath'));

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'DR_Model.mat');

testCSV = fullfile( ...
    projectRoot, ...
    'data', ...
    'test.csv');

testImageFolder = fullfile( ...
    projectRoot, ...
    'data', ...
    'test_images', ...
    'test_images');

resultsFolder = fullfile( ...
    projectRoot, ...
    'results');

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% ================================================================
% 2. CHECK FILES
% ================================================================

fprintf('Checking project files...\n');

if ~isfile(modelFile)

    error( ...
        'Model file not found:\n%s', ...
        modelFile);

end

if ~isfile(testCSV)

    error( ...
        'Test CSV not found:\n%s', ...
        testCSV);

end

if ~isfolder(testImageFolder)

    error( ...
        'Test image folder not found:\n%s', ...
        testImageFolder);

end

fprintf('All required files found.\n');

%% ================================================================
% 3. LOAD EXISTING MODEL
% ================================================================

fprintf('\n');
fprintf('Loading trained model...\n');

S = load(modelFile);

if ~isfield(S,'trainedNet')

    error( ...
        'DR_Model.mat does not contain "trainedNet".');

end

trainedNet = S.trainedNet;

if isfield(S,'classNames')

    classNames = string(S.classNames);

else

    classNames = [ ...
        "No DR", ...
        "Mild DR", ...
        "Moderate DR", ...
        "Severe DR", ...
        "Proliferative DR"];

end

fprintf('Model loaded successfully.\n');

fprintf('\nModel classes:\n');

for c = 1:numel(classNames)

    fprintf( ...
        '  %d -> %s\n', ...
        c-1, ...
        classNames(c));

end

%% ================================================================
% 4. DETERMINE INPUT SIZE
% ================================================================

try

    inputSize = trainedNet.Layers(1).InputSize;

catch

    error( ...
        ['Unable to determine network input size. ' ...
         'Inspect trainedNet manually.']);

end

fprintf('\nNetwork input size: [%d %d %d]\n', ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3));

%% ================================================================
% 5. LOAD TEST CSV
% ================================================================

fprintf('\n');
fprintf('Loading test dataset...\n');

testTable = readtable(testCSV);

fprintf('Test dataset columns:\n');

disp(testTable.Properties.VariableNames);

%% ================================================================
% 6. CHECK CSV COLUMNS
% ================================================================

if ~ismember( ...
        'id_code', ...
        testTable.Properties.VariableNames)

    error( ...
        'Column "id_code" is missing from test.csv.');

end

if ~ismember( ...
        'diagnosis', ...
        testTable.Properties.VariableNames)

    error( ...
        'Column "diagnosis" is missing from test.csv.');

end

%% ================================================================
% 7. READ TEST IMAGE IDs
% ================================================================

imageIDs = string(testTable.id_code);

numImages = numel(imageIDs);

fprintf('\n');
fprintf('Total test images: %d\n',numImages);

%% ================================================================
% 8. CONVERT CSV DIAGNOSIS TO NUMERIC 0-4
% ================================================================

rawDiagnosis = testTable.diagnosis;

diagnosisNumeric = nan(numImages,1);

%% Numeric CSV

if isnumeric(rawDiagnosis)

    diagnosisNumeric = double(rawDiagnosis);

%% String/categorical CSV

else

    diagnosisString = string(rawDiagnosis);

    diagnosisNumeric = ...
        str2double(diagnosisString);

    %% Handle text labels if present

    for i = 1:numImages

        if isnan(diagnosisNumeric(i))

            label = lower( ...
                strtrim(diagnosisString(i)));

            switch label

                case {"no dr","no_dr","nodr","0"}

                    diagnosisNumeric(i) = 0;

                case {"mild dr","mild_dr","mild","1"}

                    diagnosisNumeric(i) = 1;

                case {"moderate dr","moderate_dr","moderate","2"}

                    diagnosisNumeric(i) = 2;

                case {"severe dr","severe_dr","severe","3"}

                    diagnosisNumeric(i) = 3;

                case {"proliferative dr", ...
                      "proliferative_dr", ...
                      "proliferative", ...
                      "4"}

                    diagnosisNumeric(i) = 4;

            end

        end

    end

end

%% ================================================================
% 9. VERIFY LABELS
% ================================================================

if any(isnan(diagnosisNumeric))

    badRows = find(isnan(diagnosisNumeric));

    fprintf('\nInvalid diagnosis rows:\n');

    disp(testTable(badRows,:));

    error( ...
        'Some diagnosis values could not be converted to 0-4.');

end

if any(~ismember(diagnosisNumeric,0:4))

    invalidValues = unique( ...
        diagnosisNumeric( ...
        ~ismember(diagnosisNumeric,0:4)));

    fprintf('\nInvalid diagnosis values:\n');

    disp(invalidValues);

    error( ...
        'Diagnosis values must be between 0 and 4.');

end

%% ================================================================
% 10. MAP TRUE LABELS TO MODEL CLASS NAMES
% ================================================================

trueLabelStrings = ...
    classNames(diagnosisNumeric + 1);

trueLabels = categorical( ...
    trueLabelStrings, ...
    classNames);

%% ================================================================
% 11. DISPLAY TEST DISTRIBUTION
% ================================================================

fprintf('\n');
fprintf('Test set distribution:\n');

fprintf( ...
    '---------------------------------------------\n');

for c = 1:numel(classNames)

    countClass = sum( ...
        diagnosisNumeric == c-1);

    fprintf( ...
        '%-22s : %d images\n', ...
        classNames(c), ...
        countClass);

end

fprintf( ...
    '---------------------------------------------\n');

%% ================================================================
% 12. CREATE IMAGE FILE LIST
%
% IMPORTANT:
% The image order follows test.csv.
% This allows us to keep the true labels and predictions aligned.
% ================================================================

imageFiles = strings(numImages,1);

missingImages = false(numImages,1);

for i = 1:numImages

    imageFiles(i) = fullfile( ...
        testImageFolder, ...
        imageIDs(i) + ".png");

    if ~isfile(imageFiles(i))

        missingImages(i) = true;

    end

end

%% ================================================================
% 13. REPORT MISSING IMAGES
% ================================================================

if any(missingImages)

    fprintf('\n');
    fprintf( ...
        'WARNING: %d images are missing.\n', ...
        sum(missingImages));

    missingIDs = imageIDs(missingImages);

    disp(missingIDs);

end

%% ================================================================
% 14. REMOVE MISSING IMAGES
% ================================================================

validRows = ~missingImages;

imageFilesValid = imageFiles(validRows);

imageIDsValid = imageIDs(validRows);

trueLabelsValid = trueLabels(validRows);

diagnosisValid = diagnosisNumeric(validRows);

numValidImages = numel(imageFilesValid);

fprintf('\n');
fprintf( ...
    'Images available for validation: %d\n', ...
    numValidImages);

if numValidImages == 0

    error('No valid test images were found.');

end

%% ================================================================
% 15. CREATE IMAGE DATASTORE
%
% This is the important correction.
%
% We now use MATLAB's datastore pipeline instead of manually
% imread -> resize -> im2single -> predict.
% ================================================================

fprintf('\n');
fprintf('Creating test image datastore...\n');

imdsTest = imageDatastore( ...
    cellstr(imageFilesValid), ...
    'Labels',trueLabelsValid);

%% ================================================================
% 16. VERIFY DATASTORE
% ================================================================

fprintf('\n');
fprintf('Datastore verification:\n');

fprintf( ...
    'Number of images: %d\n', ...
    numel(imdsTest.Files));

%% ================================================================
% 17. CREATE AUGMENTED IMAGE DATASTORE
%
% Same style of preprocessing used in the training workflow:
%
%   augmentedImageDatastore
%   ColorPreprocessing = gray2rgb
%
% No random augmentation is applied to the test set.
% ================================================================

fprintf('\n');
fprintf('Preparing test datastore...\n');

augTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing','gray2rgb');

fprintf('Test datastore ready.\n');

%% ================================================================
% 18. RUN MINIBATCH PREDICTION
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('          RUNNING INDEPENDENT TEST PREDICTION\n');
fprintf('============================================================\n');
fprintf('\n');

predictionTimer = tic;

try

    scores = minibatchpredict( ...
        trainedNet, ...
        augTest);

catch ME

    fprintf('\nPrediction failed.\n');
    fprintf('%s\n',ME.message);

    error( ...
        'Test prediction could not be completed.');

end

elapsedTime = toc(predictionTimer);

fprintf('\n');
fprintf( ...
    'Prediction completed in %.2f seconds.\n', ...
    elapsedTime);

%% ================================================================
% 19. CONVERT OUTPUT TO NUMERIC ARRAY
% ================================================================

if isa(scores,'dlarray')

    scores = extractdata(scores);

end

scores = gather(scores);

scores = double(scores);

%% ================================================================
% 20. INSPECT SCORE DIMENSIONS
% ================================================================

fprintf('\n');
fprintf('Raw prediction output size:\n');

disp(size(scores));

%% ================================================================
% 21. STANDARDIZE SCORE MATRIX
%
% Desired format:
%
%       rows    = images
%       columns = five DR classes
%
%       N x 5
% ================================================================

if isvector(scores)

    error( ...
        'Prediction output unexpectedly contains only one dimension.');

end

numClasses = numel(classNames);

if size(scores,2) == numClasses

    % Already N x 5

elseif size(scores,1) == numClasses

    % Convert 5 x N -> N x 5

    scores = scores';

else

    error( ...
        ['Prediction output dimensions do not match the ' ...
         'five DR classes.']);

end

fprintf('Standardized prediction output size:\n');

disp(size(scores));

%% ================================================================
% 22. CONVERT SCORES TO PROBABILITIES
% ================================================================

probabilities = zeros( ...
    size(scores));

for i = 1:size(scores,1)

    currentScores = scores(i,:);

    %% If already probabilities

    if all(currentScores >= 0) && ...
            abs(sum(currentScores)-1) < 1e-3

        p = currentScores;

    %% Otherwise apply softmax

    else

        shifted = ...
            currentScores - max(currentScores);

        expScores = exp(shifted);

        p = expScores ./ sum(expScores);

    end

    probabilities(i,:) = p;

end

%% ================================================================
% 23. PREDICTED CLASS
% ================================================================

[maximumProbability,predictedIndex] = ...
    max(probabilities,[],2);

predictedLabelStrings = ...
    classNames(predictedIndex);

predictedLabels = categorical( ...
    predictedLabelStrings, ...
    classNames);

%% ================================================================
% 24. BASIC SCORE DIAGNOSTICS
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                   SCORE DIAGNOSTICS\n');
fprintf('============================================================\n');

fprintf('\nAverage probability by predicted class:\n');

meanProbabilities = mean(probabilities,1);

for c = 1:numClasses

    fprintf( ...
        '  %-22s : %.2f%%\n', ...
        classNames(c), ...
        meanProbabilities(c)*100);

end

fprintf('\n');

fprintf( ...
    'Average maximum confidence: %.2f%%\n', ...
    mean(maximumProbability)*100);

%% ================================================================
% 25. PREDICTED CLASS DISTRIBUTION
%
% This is extremely important because the previous validation
% appeared to classify every image as No DR.
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 PREDICTED CLASS DISTRIBUTION\n');
fprintf('============================================================\n');

predictedCounts = zeros(numClasses,1);

for c = 1:numClasses

    predictedCounts(c) = sum( ...
        predictedIndex == c);

    fprintf( ...
        '%-22s : %d images (%.2f%%)\n', ...
        classNames(c), ...
        predictedCounts(c), ...
        100*predictedCounts(c)/numValidImages);

end

%% ================================================================
% 26. CHECK FOR MODEL COLLAPSE
% ================================================================

if predictedCounts(1) == numValidImages

    fprintf('\n');
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    fprintf('WARNING: ALL TEST IMAGES ARE PREDICTED AS NO DR.\n');
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

    fprintf('\n');
    fprintf(['This means the model is currently producing only ' ...
             'the No DR class on the test set.\n']);

    fprintf('\n');
    fprintf(['We will NOT use this as a final performance claim ' ...
             'until the model behavior is investigated.\n']);

end

%% ================================================================
% 27. CONFUSION MATRIX
% ================================================================

fprintf('\n');
fprintf('Generating confusion matrix...\n');

figure( ...
    'Name', ...
    'DR Test Set Confusion Matrix', ...
    'Color', ...
    'w');

cmChart = confusionchart( ...
    trueLabelsValid, ...
    predictedLabels, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

cmChart.Title = ...
    'Diabetic Retinopathy - Independent Test Set';

%% ================================================================
% 28. NUMERIC CONFUSION MATRIX
% ================================================================

cm = confusionmat( ...
    trueLabelsValid, ...
    predictedLabels, ...
    'Order', ...
    categorical(classNames,classNames));

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    CONFUSION MATRIX\n');
fprintf('============================================================\n');

disp(cm);

%% ================================================================
% 29. OVERALL ACCURACY
% ================================================================

accuracy = mean( ...
    predictedLabels == trueLabelsValid);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 OVERALL MODEL PERFORMANCE\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf( ...
    'Accuracy: %.2f%%\n', ...
    accuracy*100);

%% ================================================================
% 30. PER-CLASS METRICS
% ================================================================

precision = zeros(numClasses,1);

sensitivity = zeros(numClasses,1);

specificity = zeros(numClasses,1);

f1Score = zeros(numClasses,1);

TP_all = zeros(numClasses,1);

TN_all = zeros(numClasses,1);

FP_all = zeros(numClasses,1);

FN_all = zeros(numClasses,1);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    PER-CLASS PERFORMANCE\n');
fprintf('============================================================\n');

for c = 1:numClasses

    TP = cm(c,c);

    FN = sum(cm(c,:)) - TP;

    FP = sum(cm(:,c)) - TP;

    TN = sum(cm(:)) - TP - FN - FP;

    TP_all(c) = TP;

    TN_all(c) = TN;

    FP_all(c) = FP;

    FN_all(c) = FN;

    %% Precision

    if TP + FP > 0

        precision(c) = ...
            TP/(TP+FP);

    else

        precision(c) = 0;

    end

    %% Sensitivity

    if TP + FN > 0

        sensitivity(c) = ...
            TP/(TP+FN);

    else

        sensitivity(c) = 0;

    end

    %% Specificity

    if TN + FP > 0

        specificity(c) = ...
            TN/(TN+FP);

    else

        specificity(c) = 0;

    end

    %% F1

    if precision(c) + sensitivity(c) > 0

        f1Score(c) = ...
            2*precision(c)*sensitivity(c) / ...
            (precision(c)+sensitivity(c));

    else

        f1Score(c) = 0;

    end

    fprintf('\n');

    fprintf( ...
        '%s\n', ...
        classNames(c));

    fprintf( ...
        '  TP           : %d\n', ...
        TP);

    fprintf( ...
        '  TN           : %d\n', ...
        TN);

    fprintf( ...
        '  FP           : %d\n', ...
        FP);

    fprintf( ...
        '  FN           : %d\n', ...
        FN);

    fprintf( ...
        '  Precision    : %.2f%%\n', ...
        precision(c)*100);

    fprintf( ...
        '  Sensitivity  : %.2f%%\n', ...
        sensitivity(c)*100);

    fprintf( ...
        '  Specificity  : %.2f%%\n', ...
        specificity(c)*100);

    fprintf( ...
        '  F1 Score     : %.2f%%\n', ...
        f1Score(c)*100);

end

%% ================================================================
% 31. MACRO AVERAGES
% ================================================================

macroPrecision = mean(precision);

macroSensitivity = mean(sensitivity);

macroSpecificity = mean(specificity);

macroF1 = mean(f1Score);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       MACRO AVERAGES\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf( ...
    'Macro Precision   : %.2f%%\n', ...
    macroPrecision*100);

fprintf( ...
    'Macro Sensitivity : %.2f%%\n', ...
    macroSensitivity*100);

fprintf( ...
    'Macro Specificity : %.2f%%\n', ...
    macroSpecificity*100);

fprintf( ...
    'Macro F1 Score    : %.2f%%\n', ...
    macroF1*100);

%% ================================================================
% 32. REFERABLE DR
%
% Grade >= 2:
%
%   Moderate
%   Severe
%   Proliferative
%
% Grade 0-1:
%
%   Non-referable
%
% ================================================================

actualReferable = ...
    diagnosisValid >= 2;

predictedReferable = ...
    predictedIndex >= 3;

%% Referable confusion values

referableTP = sum( ...
    actualReferable & predictedReferable);

referableTN = sum( ...
    ~actualReferable & ~predictedReferable);

referableFP = sum( ...
    ~actualReferable & predictedReferable);

referableFN = sum( ...
    actualReferable & ~predictedReferable);

%% ================================================================
% 33. REFERABLE SENSITIVITY
% ================================================================

if referableTP + referableFN > 0

    referableSensitivity = ...
        referableTP / ...
        (referableTP + referableFN);

else

    referableSensitivity = 0;

end

%% ================================================================
% 34. REFERABLE SPECIFICITY
% ================================================================

if referableTN + referableFP > 0

    referableSpecificity = ...
        referableTN / ...
        (referableTN + referableFP);

else

    referableSpecificity = 0;

end

%% ================================================================
% 35. REFERABLE PRECISION
% ================================================================

if referableTP + referableFP > 0

    referablePrecision = ...
        referableTP / ...
        (referableTP + referableFP);

else

    referablePrecision = 0;

end

%% ================================================================
% 36. REFERABLE F1
% ================================================================

if referablePrecision + referableSensitivity > 0

    referableF1 = ...
        2 * ...
        referablePrecision * ...
        referableSensitivity / ...
        (referablePrecision + referableSensitivity);

else

    referableF1 = 0;

end

%% ================================================================
% 37. DISPLAY REFERABLE DR PERFORMANCE
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              REFERABLE DR PERFORMANCE (GRADE >= 2)\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf( ...
    'True Positives  : %d\n', ...
    referableTP);

fprintf( ...
    'True Negatives  : %d\n', ...
    referableTN);

fprintf( ...
    'False Positives : %d\n', ...
    referableFP);

fprintf( ...
    'False Negatives : %d\n', ...
    referableFN);

fprintf('\n');

fprintf( ...
    'Sensitivity : %.2f%%\n', ...
    referableSensitivity*100);

fprintf( ...
    'Specificity : %.2f%%\n', ...
    referableSpecificity*100);

fprintf( ...
    'Precision   : %.2f%%\n', ...
    referablePrecision*100);

fprintf( ...
    'F1 Score    : %.2f%%\n', ...
    referableF1*100);

%% ================================================================
% 38. TARGET CHECK
% ================================================================

targetSensitivity = 0.90;

targetSpecificity = 0.85;

sensitivityTargetPassed = ...
    referableSensitivity > targetSensitivity;

specificityTargetPassed = ...
    referableSpecificity > targetSpecificity;

fprintf('\n');
fprintf('============================================================\n');
fprintf('                         TARGET CHECK\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf( ...
    'Target sensitivity : >90%%\n');

fprintf( ...
    'Actual sensitivity : %.2f%%\n', ...
    referableSensitivity*100);

if sensitivityTargetPassed

    fprintf( ...
        'STATUS              : PASSED\n');

else

    fprintf( ...
        'STATUS              : NOT YET MET\n');

end

fprintf('\n');

fprintf( ...
    'Target specificity : >85%%\n');

fprintf( ...
    'Actual specificity : %.2f%%\n', ...
    referableSpecificity*100);

if specificityTargetPassed

    fprintf( ...
        'STATUS              : PASSED\n');

else

    fprintf( ...
        'STATUS              : NOT YET MET\n');

end

%% ================================================================
% 39. PERFORMANCE TABLE
% ================================================================

performanceTable = table( ...
    classNames(:), ...
    precision, ...
    sensitivity, ...
    specificity, ...
    f1Score, ...
    TP_all, ...
    TN_all, ...
    FP_all, ...
    FN_all, ...
    'VariableNames', ...
    { ...
    'Class', ...
    'Precision', ...
    'Sensitivity', ...
    'Specificity', ...
    'F1Score', ...
    'TP', ...
    'TN', ...
    'FP', ...
    'FN'});

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    PERFORMANCE TABLE\n');
fprintf('============================================================\n');

disp(performanceTable);

%% ================================================================
% 40. PER-IMAGE RESULTS TABLE
% ================================================================

resultTable = table( ...
    imageIDsValid, ...
    diagnosisValid, ...
    trueLabelStrings(validRows), ...
    predictedIndex, ...
    predictedLabelStrings, ...
    maximumProbability, ...
    'VariableNames', ...
    { ...
    'ImageID', ...
    'TrueGrade', ...
    'TrueClass', ...
    'PredictedGrade', ...
    'PredictedClass', ...
    'Confidence'});

%% Add probability columns

for c = 1:numClasses

    safeName = matlab.lang.makeValidName( ...
        classNames(c));

    resultTable.( ...
        safeName + "_Probability") = ...
        probabilities(:,c);

end

%% ================================================================
% 41. SAVE COMPLETE VALIDATION STRUCTURE
% ================================================================

validationResults = struct();

%% Project information

validationResults.projectRoot = ...
    projectRoot;

validationResults.modelFile = ...
    modelFile;

validationResults.testCSV = ...
    testCSV;

validationResults.testImageFolder = ...
    testImageFolder;

%% Dataset information

validationResults.numTestImages = ...
    numImages;

validationResults.numValidImages = ...
    numValidImages;

validationResults.numMissingImages = ...
    sum(missingImages);

%% Model information

validationResults.classNames = ...
    classNames;

validationResults.inputSize = ...
    inputSize;

%% Overall performance

validationResults.accuracy = ...
    accuracy;

%% Per-class metrics

validationResults.performanceTable = ...
    performanceTable;

validationResults.precision = ...
    precision;

validationResults.sensitivity = ...
    sensitivity;

validationResults.specificity = ...
    specificity;

validationResults.f1Score = ...
    f1Score;

%% Macro averages

validationResults.macroPrecision = ...
    macroPrecision;

validationResults.macroSensitivity = ...
    macroSensitivity;

validationResults.macroSpecificity = ...
    macroSpecificity;

validationResults.macroF1 = ...
    macroF1;

%% Referable DR

validationResults.referableTP = ...
    referableTP;

validationResults.referableTN = ...
    referableTN;

validationResults.referableFP = ...
    referableFP;

validationResults.referableFN = ...
    referableFN;

validationResults.referableSensitivity = ...
    referableSensitivity;

validationResults.referableSpecificity = ...
    referableSpecificity;

validationResults.referablePrecision = ...
    referablePrecision;

validationResults.referableF1 = ...
    referableF1;

%% Targets

validationResults.targetSensitivity = ...
    targetSensitivity;

validationResults.targetSpecificity = ...
    targetSpecificity;

validationResults.sensitivityTargetPassed = ...
    sensitivityTargetPassed;

validationResults.specificityTargetPassed = ...
    specificityTargetPassed;

%% Predictions

validationResults.imageIDs = ...
    imageIDsValid;

validationResults.trueLabels = ...
    trueLabelsValid;

validationResults.predictedLabels = ...
    predictedLabels;

validationResults.trueGrades = ...
    diagnosisValid;

validationResults.predictedGrades = ...
    predictedIndex - 1;

validationResults.probabilities = ...
    probabilities;

validationResults.confidence = ...
    maximumProbability;

validationResults.confusionMatrix = ...
    cm;

%% ================================================================
% 42. SAVE MAT FILE
% ================================================================

validationMAT = fullfile( ...
    resultsFolder, ...
    'validationResults.mat');

save( ...
    validationMAT, ...
    'validationResults');

%% ================================================================
% 43. SAVE PERFORMANCE CSV
% ================================================================

performanceCSV = fullfile( ...
    resultsFolder, ...
    'DR_Performance_Table.csv');

writetable( ...
    performanceTable, ...
    performanceCSV);

%% ================================================================
% 44. SAVE PER-IMAGE PREDICTIONS
% ================================================================

predictionCSV = fullfile( ...
    resultsFolder, ...
    'DR_Test_Predictions.csv');

writetable( ...
    resultTable, ...
    predictionCSV);

%% ================================================================
% 45. FINAL SUMMARY
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    VALIDATION COMPLETE\n');
fprintf('============================================================\n');

fprintf('\n');

fprintf( ...
    'Test images              : %d\n', ...
    numImages);

fprintf( ...
    'Valid images             : %d\n', ...
    numValidImages);

fprintf( ...
    'Missing images           : %d\n', ...
    sum(missingImages));

fprintf('\n');

fprintf( ...
    'Overall accuracy         : %.2f%%\n', ...
    accuracy*100);

fprintf('\n');

fprintf( ...
    'Referable DR sensitivity : %.2f%%\n', ...
    referableSensitivity*100);

fprintf( ...
    'Referable DR specificity : %.2f%%\n', ...
    referableSpecificity*100);

fprintf('\n');

fprintf( ...
    'Validation MAT:\n%s\n', ...
    validationMAT);

fprintf('\n');

fprintf( ...
    'Performance CSV:\n%s\n', ...
    performanceCSV);

fprintf('\n');

fprintf( ...
    'Per-image predictions:\n%s\n', ...
    predictionCSV);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    END OF VALIDATION\n');
fprintf('============================================================\n');
fprintf('\n');

end