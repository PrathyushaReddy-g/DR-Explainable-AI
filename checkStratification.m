clc;
clear;

% Project data folder
dataFolder = fullfile(pwd, 'data');

% CSV files
trainFile = fullfile(dataFolder, 'train_1.csv');
validFile = fullfile(dataFolder, 'valid.csv');
testFile  = fullfile(dataFolder, 'test.csv');

% Read CSV files
trainTbl = readtable(trainFile);
validTbl = readtable(validFile);
testTbl  = readtable(testFile);

% Convert diagnosis to categorical
trainLabels = categorical(trainTbl.diagnosis);
validLabels = categorical(validTbl.diagnosis);
testLabels  = categorical(testTbl.diagnosis);

% Class names
classes = categorical(0:4);

fprintf('\n============================================\n');
fprintf('       DATASET STRATIFICATION CHECK\n');
fprintf('============================================\n\n');

% Display each split
checkSplit('TRAIN', trainLabels, classes);
checkSplit('VALIDATION', validLabels, classes);
checkSplit('TEST', testLabels, classes);

% Overall distribution
allLabels = [trainLabels; validLabels; testLabels];

fprintf('\n============================================\n');
fprintf('          OVERALL DISTRIBUTION\n');
fprintf('============================================\n');

for i = 1:numel(classes)
    count = sum(allLabels == classes(i));
    percentage = 100 * count / numel(allLabels);

    fprintf('Grade %d : %4d images  (%6.2f%%)\n', ...
        double(classes(i)), count, percentage);
end

fprintf('\nTotal images: %d\n', numel(allLabels));

fprintf('\n============================================\n');
fprintf('       STRATIFICATION INTERPRETATION\n');
fprintf('============================================\n\n');

fprintf(['Compare the percentages of Grade 0-4 across TRAIN, VALIDATION,\n' ...
    'and TEST.\n\n']);

fprintf(['If the percentages are reasonably similar, your existing\n' ...
    'splits are already approximately stratified.\n\n']);

fprintf(['If Grade 3 or Grade 4 has a much smaller percentage in TRAIN\n' ...
    'than in TEST, we should create a better stratified split.\n\n']);

fprintf('============================================\n');


function checkSplit(name, labels, classes)

fprintf('\n--------------------------------------------\n');
fprintf('%s SET\n', name);
fprintf('Total images: %d\n', numel(labels));
fprintf('--------------------------------------------\n');

for i = 1:numel(classes)

    count = sum(labels == classes(i));
    percentage = 100 * count / numel(labels);

    fprintf('Grade %d : %4d images  (%6.2f%%)\n', ...
        double(classes(i)), count, percentage);
end
end