function trainDRModel
% TRAIN DR MODEL
% Explainable AI for Diabetic Retinopathy
%
% 5-class classification:
%   0 = No DR
%   1 = Mild
%   2 = Moderate
%   3 = Severe
%   4 = Proliferative
%
% Dataset:
%   data/train_1.csv
%   data/valid.csv
%   data/test.csv
%
% Test data is NEVER used for training.
%
% Model:
%   ResNet-18
%
% Improvements:
%   - Class-balanced loss
%   - Training augmentation
%   - Fixed train/validation/test split
%   - Final independent test evaluation
%   - Per-class recall
%   - Referable DR evaluation

clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       DIABETIC RETINOPATHY - RESNET-18 TRAINING\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. PROJECT PATHS
% =============================================================

projectFolder = pwd;

dataFolder = fullfile(projectFolder,'data');
modelFolder = fullfile(projectFolder,'models');
resultsFolder = fullfile(projectFolder,'results');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

trainCSV = fullfile(dataFolder,'train_1.csv');
validCSV = fullfile(dataFolder,'valid.csv');
testCSV  = fullfile(dataFolder,'test.csv');

fprintf('Project folder:\n%s\n\n',projectFolder);

%% ============================================================
% 2. CHECK DATASET FILES
% =============================================================

if ~isfile(trainCSV)
    error('Training CSV not found:\n%s',trainCSV);
end

if ~isfile(validCSV)
    error('Validation CSV not found:\n%s',validCSV);
end

if ~isfile(testCSV)
    error('Test CSV not found:\n%s',testCSV);
end

fprintf('Dataset files found successfully.\n\n');

%% ============================================================
% 3. READ CSV FILES
% =============================================================

trainTbl = readtable(trainCSV);
validTbl = readtable(validCSV);
testTbl  = readtable(testCSV);

fprintf('Training images   : %d\n',height(trainTbl));
fprintf('Validation images : %d\n',height(validTbl));
fprintf('Test images       : %d\n\n',height(testTbl));

%% ============================================================
% 4. CHECK REQUIRED COLUMNS
% =============================================================

requiredColumns = {'id_code','diagnosis'};

for k = 1:numel(requiredColumns)

    if ~ismember(requiredColumns{k}, ...
            trainTbl.Properties.VariableNames)

        error('Column "%s" missing from train_1.csv.', ...
            requiredColumns{k});
    end

    if ~ismember(requiredColumns{k}, ...
            validTbl.Properties.VariableNames)

        error('Column "%s" missing from valid.csv.', ...
            requiredColumns{k});
    end

    if ~ismember(requiredColumns{k}, ...
            testTbl.Properties.VariableNames)

        error('Column "%s" missing from test.csv.', ...
            requiredColumns{k});
    end
end

%% ============================================================
% 5. READ DIAGNOSIS VALUES
% =============================================================

trainDiagnosis = double(trainTbl.diagnosis);
validDiagnosis = double(validTbl.diagnosis);
testDiagnosis  = double(testTbl.diagnosis);

% Make sure labels are exactly 0-4.

if ~all(ismember(trainDiagnosis,0:4))
    error('Training labels must contain only values 0,1,2,3,4.');
end

if ~all(ismember(validDiagnosis,0:4))
    error('Validation labels must contain only values 0,1,2,3,4.');
end

if ~all(ismember(testDiagnosis,0:4))
    error('Test labels must contain only values 0,1,2,3,4.');
end

%% ============================================================
% 6. DEFINE DR CLASS NAMES
% =============================================================

classNames = categorical( ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'}, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

numClasses = 5;

%% ============================================================
% 7. FIND FUNDUS IMAGES
% =============================================================

fprintf('Searching for fundus images...\n');

imageFiles = findFundusImages(dataFolder);

if isempty(imageFiles)

    error(['No images were found inside:\n%s\n\n' ...
           'Check your data folder.'],dataFolder);

end

fprintf('Images found: %d\n\n',numel(imageFiles));

%% ============================================================
% 8. CREATE IMAGE LOOKUP
% =============================================================

fprintf('Creating image lookup table...\n');

imageMap = containers.Map( ...
    'KeyType','char', ...
    'ValueType','char');

for k = 1:numel(imageFiles)

    [~,fileName,~] = fileparts(imageFiles{k});

    key = lower(strtrim(fileName));

    imageMap(key) = imageFiles{k};

end

fprintf('Image lookup created.\n\n');

%% ============================================================
% 9. MATCH TRAINING IMAGES
% =============================================================

fprintf('Matching training images...\n');

trainPaths = matchImages( ...
    trainTbl.id_code, ...
    imageMap);

%% ============================================================
% 10. MATCH VALIDATION IMAGES
% =============================================================

fprintf('Matching validation images...\n');

validPaths = matchImages( ...
    validTbl.id_code, ...
    imageMap);

%% ============================================================
% 11. MATCH TEST IMAGES
% =============================================================

fprintf('Matching test images...\n');

testPaths = matchImages( ...
    testTbl.id_code, ...
    imageMap);

fprintf('\nAll CSV entries matched with images.\n\n');

%% ============================================================
% 12. CREATE CATEGORICAL LABELS
% =============================================================

trainLabels = categorical( ...
    trainDiagnosis, ...
    0:4, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

validLabels = categorical( ...
    validDiagnosis, ...
    0:4, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

testLabels = categorical( ...
    testDiagnosis, ...
    0:4, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

%% ============================================================
% 13. CREATE IMAGE DATASTORES
% =============================================================

imdsTrain = imageDatastore( ...
    trainPaths, ...
    'Labels',trainLabels);

imdsValidation = imageDatastore( ...
    validPaths, ...
    'Labels',validLabels);

imdsTest = imageDatastore( ...
    testPaths, ...
    'Labels',testLabels);

%% ============================================================
% 14. DISPLAY CLASS DISTRIBUTION
% =============================================================

fprintf('============================================================\n');
fprintf('              CLASS DISTRIBUTION\n');
fprintf('============================================================\n');

displayClassDistribution( ...
    'TRAINING', ...
    trainLabels);

displayClassDistribution( ...
    'VALIDATION', ...
    validLabels);

displayClassDistribution( ...
    'TEST', ...
    testLabels);

%% ============================================================
% 15. CALCULATE CLASS WEIGHTS
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              CLASS BALANCING\n');
fprintf('============================================================\n');

classCounts = zeros(numClasses,1);

for c = 0:4

    classCounts(c+1) = ...
        sum(trainDiagnosis == c);

end

fprintf('\nOriginal training counts:\n\n');

for c = 1:numClasses

    fprintf('%-15s : %4d\n', ...
        char(classNames(c)), ...
        classCounts(c));

end

% ------------------------------------------------------------
% Inverse-frequency class weights
% ------------------------------------------------------------

totalTraining = sum(classCounts);

classWeights = ...
    totalTraining ./ ...
    (numClasses .* classCounts);

% Normalize weights.
classWeights = ...
    classWeights ./ mean(classWeights);

fprintf('\nCalculated class weights:\n\n');

for c = 1:numClasses

    fprintf('%-15s : %.3f\n', ...
        char(classNames(c)), ...
        classWeights(c));

end

%% ============================================================
% 16. LOAD RESNET-18
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              LOADING RESNET-18\n');
fprintf('============================================================\n\n');

fprintf('Loading pretrained ResNet-18...\n');

% MATLAB R2026a:
% For transfer learning with NumClasses specified,
% imagePretrainedNetwork returns ONE output.

net = imagePretrainedNetwork( ...
    'resnet18', ...
    'NumClasses',numClasses);

% Use our explicit DR class ordering.
classNamesNetwork = classNames;

fprintf('ResNet-18 loaded successfully.\n');

% Input size.
inputSize = net.Layers(1).InputSize;

fprintf('Network input size: %d x %d x %d\n', ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3));

%% ============================================================
% 17. CREATE DATA AUGMENTATION
% =============================================================

fprintf('\nCreating training augmentation...\n');

augmenter = imageDataAugmenter( ...
    'RandRotation',[-15 15], ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXScale',[0.90 1.10], ...
    'RandYScale',[0.90 1.10]);

%% ============================================================
% 18. AUGMENTED DATASTORES
% =============================================================

augimdsTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    'DataAugmentation',augmenter, ...
    'ColorPreprocessing','gray2rgb');

% Validation is NOT augmented.
augimdsValidation = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsValidation, ...
    'ColorPreprocessing','gray2rgb');

% Test is NOT augmented.
augimdsTest = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTest, ...
    'ColorPreprocessing','gray2rgb');

fprintf('Training augmentation created.\n');
fprintf('Validation augmentation: NONE\n');
fprintf('Test augmentation: NONE\n');

%% ============================================================
% 19. CREATE WEIGHTED LOSS
% =============================================================

fprintf('\nCreating class-weighted loss function...\n');

lossFcn = @(Y,T) ...
    weightedCrossEntropy(Y,T,classWeights);

fprintf('Weighted cross-entropy ready.\n');

%% ============================================================
% 20. TRAINING OPTIONS
% =============================================================

fprintf('\nCreating training options...\n');

miniBatchSize = 16;

validationFrequency = max( ...
    1, ...
    floor(numel(trainLabels)/miniBatchSize));

options = trainingOptions( ...
    'adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'L2Regularization',1e-4, ...
    'GradientThreshold',1, ...
    'Plots','training-progress', ...
    'Verbose',true);

%% ============================================================
% 21. START TRAINING
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  STARTING TRAINING\n');
fprintf('============================================================\n\n');

fprintf('Model              : ResNet-18\n');
fprintf('Training images    : %d\n',numel(trainLabels));
fprintf('Validation images  : %d\n',numel(validLabels));
fprintf('Test images        : %d\n',numel(testLabels));
fprintf('Epochs             : %d\n',20);
fprintf('Mini-batch size    : %d\n',miniBatchSize);
fprintf('Initial LR         : %.1e\n',1e-4);
fprintf('Class balancing    : YES\n');
fprintf('Training augment.  : YES\n');
fprintf('Validation augment.: NO\n');
fprintf('Test augment.      : NO\n');
fprintf('\n');

fprintf('Test set will NOT be used during training.\n\n');

[trainedNet,info] = trainnet( ...
    augimdsTrain, ...
    net, ...
    lossFcn, ...
    options);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  TRAINING FINISHED\n');
fprintf('============================================================\n');

%% ============================================================
% 22. VALIDATION PREDICTIONS
% =============================================================

fprintf('\nGenerating validation predictions...\n');

validationScores = minibatchpredict( ...
    trainedNet, ...
    augimdsValidation);

validationScores = ...
    gather(extractdata(validationScores));

validationScores = squeeze(validationScores);

% Make sure dimensions are samples x classes.
if size(validationScores,2) ~= numClasses && ...
        size(validationScores,1) == numClasses

    validationScores = validationScores';

end

[~,validationIndex] = ...
    max(validationScores,[],2);

validationPredicted = categorical( ...
    validationIndex-1, ...
    0:4, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

%% ============================================================
% 23. VALIDATION ACCURACY
% =============================================================

validationAccuracy = mean( ...
    validationPredicted == validLabels);

fprintf('\nValidation Accuracy: %.2f %%\n', ...
    100*validationAccuracy);

%% ============================================================
% 24. VALIDATION CONFUSION MATRIX
% =============================================================

validationCM = confusionmat( ...
    validLabels, ...
    validationPredicted, ...
    'Order',classNames);

fprintf('\nValidation confusion matrix:\n');

disp(validationCM);

fprintf('\nValidation per-class recall:\n');

displayRecall( ...
    validationCM, ...
    classNames);

%% ============================================================
% 25. FINAL TEST PREDICTIONS
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             FINAL TEST EVALUATION\n');
fprintf('============================================================\n\n');

fprintf('IMPORTANT:\n');
fprintf('The test set was not used for training.\n');
fprintf('The test set is being evaluated only now.\n\n');

testScores = minibatchpredict( ...
    trainedNet, ...
    augimdsTest);

testScores = ...
    gather(extractdata(testScores));

testScores = squeeze(testScores);

if size(testScores,2) ~= numClasses && ...
        size(testScores,1) == numClasses

    testScores = testScores';

end

[~,testIndex] = ...
    max(testScores,[],2);

testPredicted = categorical( ...
    testIndex-1, ...
    0:4, ...
    {'No_DR','Mild','Moderate','Severe','Proliferative'});

%% ============================================================
% 26. TEST ACCURACY
% =============================================================

testAccuracy = mean( ...
    testPredicted == testLabels);

fprintf('\n');
fprintf('TEST ACCURACY: %.2f %%\n', ...
    100*testAccuracy);

%% ============================================================
% 27. TEST CONFUSION MATRIX
% =============================================================

testCM = confusionmat( ...
    testLabels, ...
    testPredicted, ...
    'Order',classNames);

fprintf('\nTest confusion matrix:\n');

disp(testCM);

%% ============================================================
% 28. PER-CLASS RECALL
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('              PER-CLASS RECALL\n');
fprintf('============================================================\n');

displayRecall( ...
    testCM, ...
    classNames);

%% ============================================================
% 29. REFERABLE DR
% ============================================================

% Referable DR:
%
%   0 = No DR       -> Non-referable
%   1 = Mild        -> Non-referable
%   2 = Moderate    -> Referable
%   3 = Severe      -> Referable
%   4 = Proliferative -> Referable

actualReferable = ...
    testDiagnosis >= 2;

predictedReferable = ...
    (testIndex-1) >= 2;

TP = sum( ...
    actualReferable & ...
    predictedReferable);

TN = sum( ...
    ~actualReferable & ...
    ~predictedReferable);

FP = sum( ...
    ~actualReferable & ...
    predictedReferable);

FN = sum( ...
    actualReferable & ...
    ~predictedReferable);

referableSensitivity = ...
    TP / max(TP+FN,eps);

referableSpecificity = ...
    TN / max(TN+FP,eps);

referablePPV = ...
    TP / max(TP+FP,eps);

referableNPV = ...
    TN / max(TN+FN,eps);

fprintf('\n');
fprintf('============================================================\n');
fprintf('           REFERABLE DR PERFORMANCE\n');
fprintf('============================================================\n\n');

fprintf('Sensitivity : %.2f %%\n', ...
    100*referableSensitivity);

fprintf('Specificity : %.2f %%\n', ...
    100*referableSpecificity);

fprintf('PPV         : %.2f %%\n', ...
    100*referablePPV);

fprintf('NPV         : %.2f %%\n', ...
    100*referableNPV);

fprintf('\nBinary confusion matrix:\n');

binaryCM = [TN FP; FN TP];

disp(binaryCM);

%% ============================================================
% 30. CREATE TEST CONFUSION MATRIX FIGURE
% =============================================================

fprintf('\nCreating confusion matrix figure...\n');

fig = figure( ...
    'Name','DR Test Confusion Matrix', ...
    'NumberTitle','off');

confusionchart( ...
    testLabels, ...
    testPredicted, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

title(sprintf( ...
    'ResNet-18 Test Confusion Matrix - Accuracy %.2f%%', ...
    100*testAccuracy));

saveas( ...
    fig, ...
    fullfile( ...
    resultsFolder, ...
    'DR_Test_Confusion_Matrix.png'));

%% ============================================================
% 31. SAVE TEST PREDICTIONS
% =============================================================

fprintf('Saving test predictions...\n');

testConfidence = ...
    max(testScores,[],2);

predictionTable = table( ...
    testTbl.id_code, ...
    testDiagnosis, ...
    testIndex-1, ...
    testConfidence, ...
    'VariableNames',{ ...
    'id_code', ...
    'actual_diagnosis', ...
    'predicted_diagnosis', ...
    'confidence'});

writetable( ...
    predictionTable, ...
    fullfile( ...
    resultsFolder, ...
    'DR_Test_Predictions.csv'));

%% ============================================================
% 32. SAVE MODEL
% =============================================================

fprintf('\nSaving trained model...\n');

modelFile = fullfile( ...
    modelFolder, ...
    'DR_Model.mat');

save( ...
    modelFile, ...
    'trainedNet', ...
    'classNames', ...
    'classNamesNetwork', ...
    'classWeights', ...
    'classCounts', ...
    'info', ...
    'testAccuracy', ...
    'validationAccuracy', ...
    'testCM', ...
    'binaryCM', ...
    'referableSensitivity', ...
    'referableSpecificity', ...
    'referablePPV', ...
    'referableNPV', ...
    '-v7.3');

fprintf('\nMODEL SAVED:\n');
fprintf('%s\n',modelFile);

%% ============================================================
% 33. FINAL SUMMARY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  TRAINING SUMMARY\n');
fprintf('============================================================\n\n');

fprintf('Model                 : ResNet-18\n');
fprintf('Training images       : %d\n',height(trainTbl));
fprintf('Validation images     : %d\n',height(validTbl));
fprintf('Test images           : %d\n',height(testTbl));

fprintf('\nValidation Accuracy   : %.2f %%\n', ...
    100*validationAccuracy);

fprintf('Test Accuracy         : %.2f %%\n', ...
    100*testAccuracy);

fprintf('\nReferable Sensitivity : %.2f %%\n', ...
    100*referableSensitivity);

fprintf('Referable Specificity : %.2f %%\n', ...
    100*referableSpecificity);

fprintf('\nResults folder:\n%s\n',resultsFolder);

fprintf('\nModel file:\n%s\n',modelFile);

fprintf('\n============================================================\n');
fprintf('                    TRAINING COMPLETE\n');
fprintf('============================================================\n\n');

end


%% ========================================================================
% FIND FUNDUS IMAGES
% ========================================================================

function imageFiles = findFundusImages(dataFolder)

extensions = { ...
    '*.jpg', ...
    '*.jpeg', ...
    '*.png', ...
    '*.tif', ...
    '*.tiff'};

imageFiles = {};

for e = 1:numel(extensions)

    files = dir(fullfile( ...
        dataFolder, ...
        '**', ...
        extensions{e}));

    for k = 1:numel(files)

        if ~files(k).isdir

            imageFiles{end+1,1} = ...
                fullfile( ...
                files(k).folder, ...
                files(k).name);

        end

    end

end

if ~isempty(imageFiles)

    imageFiles = unique(imageFiles);

end

end


%% ========================================================================
% MATCH CSV IDS TO IMAGE FILES
% ========================================================================

function paths = matchImages(ids,imageMap)

numImages = numel(ids);

paths = cell(numImages,1);

missing = {};

for k = 1:numImages

    id = string(ids(k));

    id = strtrim(id);

    [~,idWithoutExt,~] = ...
        fileparts(id);

    key = lower(idWithoutExt);

    if isKey(imageMap,key)

        paths{k} = imageMap(key);

    else

        missing{end+1,1} = char(id);

    end

end

if ~isempty(missing)

    fprintf('\n');
    fprintf('WARNING: %d images could not be matched.\n', ...
        numel(missing));

    fprintf('\nFirst missing IDs:\n');

    for k = 1:min(10,numel(missing))

        fprintf('  %s\n',missing{k});

    end

    error( ...
        ['Some CSV image IDs could not be matched ' ...
         'to fundus images.']);

end

end


%% ========================================================================
% DISPLAY CLASS DISTRIBUTION
% ========================================================================

function displayClassDistribution(name,labels)

fprintf('\n%s SET\n',name);
fprintf('--------------------------------------------\n');

classList = categories(labels);

for c = 1:numel(classList)

    count = sum(labels == classList{c});

    percentage = ...
        100 * count / numel(labels);

    fprintf('%-15s : %4d images (%6.2f%%)\n', ...
        classList{c}, ...
        count, ...
        percentage);

end

end


%% ========================================================================
% WEIGHTED CROSS ENTROPY
% ========================================================================

function loss = weightedCrossEntropy(Y,T,classWeights)

% ------------------------------------------------------------
% Convert network output to probabilities.
% ------------------------------------------------------------

P = softmax(Y);

% ------------------------------------------------------------
% Convert target to numeric data if necessary.
% ------------------------------------------------------------

if isa(T,'dlarray')

    Tdata = extractdata(T);

else

    Tdata = T;

end

% ------------------------------------------------------------
% Determine target format.
% ------------------------------------------------------------

if iscategorical(Tdata)

    targetIndex = double(Tdata);

elseif isvector(Tdata)

    targetIndex = double(Tdata);

else

    % One-hot encoded targets.
    [~,targetIndex] = max(Tdata,[],1);

    targetIndex = double(targetIndex);

end

targetIndex = targetIndex(:);

% ------------------------------------------------------------
% Make prediction dimensions:
%
%       Classes x Batch
% ------------------------------------------------------------

if size(P,2) ~= numel(targetIndex) && ...
        size(P,1) == numel(targetIndex)

    P = P';

end

batchSize = numel(targetIndex);

% ------------------------------------------------------------
% Calculate weighted loss.
% ------------------------------------------------------------

weightedLoss = zeros( ...
    1, ...
    batchSize, ...
    'like',P);

for n = 1:batchSize

    classIndex = targetIndex(n);

    if classIndex >= 1 && ...
            classIndex <= numel(classWeights)

        probability = P(classIndex,n);

        weight = classWeights(classIndex);

        weightedLoss(n) = ...
            -weight .* ...
            log(probability + eps);

    end

end

loss = mean(weightedLoss);

end


%% ========================================================================
% DISPLAY PER-CLASS RECALL
% ========================================================================

function displayRecall(cm,classNames)

for c = 1:size(cm,1)

    actualCount = sum(cm(c,:));

    recall = ...
        cm(c,c) / max(actualCount,eps);

    fprintf( ...
        '%-15s : %6.2f %%   (%d / %d)\n', ...
        char(classNames(c)), ...
        100*recall, ...
        cm(c,c), ...
        actualCount);

end

end