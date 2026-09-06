%% EXPLAINABLE DR SCREENING
% Diabetic Retinopathy classification + Grad-CAM
% MATLAB R2026a

clear;
clc;
close all;

%% 1. PROJECT PATH

projectRoot = 'C:\Users\LENOVO\Documents\MATLAB\DR_Explainable_AI_New';

modelFile = fullfile(projectRoot,'models','DR_Model.mat');

%% 2. LOAD TRAINED MODEL

fprintf('\n========================================\n');
fprintf('EXPLAINABLE DR SCREENING\n');
fprintf('========================================\n');

fprintf('\nLoading trained model...\n');

load(modelFile,'trainedNet','classNames');

fprintf('Model loaded successfully.\n');

%% 3. SELECT RETINAL IMAGE

fprintf('\nSelect a retinal image...\n');

[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg','Retinal Images (*.png, *.jpg, *.jpeg)'}, ...
    'Select Fundus Image');

if isequal(fileName,0)
    fprintf('No image selected.\n');
    return;
end

imageFile = fullfile(filePath,fileName);

fprintf('Selected image: %s\n',fileName);

%% 4. READ IMAGE

originalImage = imread(imageFile);

%% 5. PREPROCESS IMAGE

inputSize = [224 224];

if size(originalImage,3) == 1
    originalImage = cat(3,originalImage, ...
                           originalImage, ...
                           originalImage);
end

inputImage = imresize(originalImage,inputSize);

inputImage = im2single(inputImage);

%% 6. PREDICT DR CLASS

scores = predict(trainedNet,inputImage);

if isa(scores,'dlarray')
    scores = extractdata(scores);
end

scores = gather(scores);

scores = squeeze(scores);

% Convert scores to probabilities if necessary
scores = exp(scores - max(scores));
scores = scores ./ sum(scores);

[confidence,classIndex] = max(scores);

predictedClass = classNames(classIndex);

%% 7. DISPLAY PREDICTION

fprintf('\n========================================\n');
fprintf('DR PREDICTION\n');
fprintf('========================================\n');

fprintf('Predicted Class : %s\n',predictedClass);
fprintf('Confidence      : %.2f%%\n',confidence*100);

fprintf('\nClass probabilities:\n');

for i = 1:numel(classNames)
    fprintf('  %-20s %.2f%%\n', ...
        classNames(i),scores(i)*100);
end

%% 8. GRAD-CAM

fprintf('\nGenerating Grad-CAM explanation...\n');

try

    scoreMap = gradCAM( ...
        trainedNet, ...
        inputImage, ...
        classIndex);

catch ME

    fprintf('\nAutomatic Grad-CAM failed.\n');
    fprintf('Trying automatic feature-layer selection...\n');

    [scoreMap,featureLayer,reductionLayer] = gradCAM( ...
        trainedNet, ...
        inputImage, ...
        classIndex);

    fprintf('Feature layer: %s\n',string(featureLayer));
    fprintf('Reduction layer: %s\n',string(reductionLayer));

end

%% 9. RESIZE GRAD-CAM MAP

scoreMap = imresize(scoreMap, ...
    [size(originalImage,1),size(originalImage,2)]);

%% 10. DISPLAY RESULTS

figure( ...
    'Name','Explainable DR Screening', ...
    'NumberTitle','off', ...
    'Color','white');

tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

%% Original Image

nexttile;

imshow(originalImage);

title('Original Fundus Image', ...
    'FontSize',14, ...
    'FontWeight','bold');

%% Grad-CAM

nexttile;

imagesc(scoreMap);

axis image off;

title('Grad-CAM Attention', ...
    'FontSize',14, ...
    'FontWeight','bold');

colorbar;

%% Overlay

nexttile;

imshow(originalImage);

hold on;

imagesc(scoreMap, ...
    'AlphaData',0.45);

axis image off;

title('Explainable Prediction', ...
    'FontSize',14, ...
    'FontWeight','bold');

hold off;

%% 11. CLINICAL INTERPRETATION

fprintf('\n========================================\n');
fprintf('CLINICAL INTERPRETATION\n');
fprintf('========================================\n');

switch string(predictedClass)

    case "No DR"

        recommendation = ...
            "No obvious diabetic retinopathy detected.";

    case "Mild DR"

        recommendation = ...
            "Mild diabetic retinopathy detected. Clinical follow-up recommended.";

    case "Moderate DR"

        recommendation = ...
            "Referable diabetic retinopathy detected. Ophthalmology review recommended.";

    case "Severe DR"

        recommendation = ...
            "Referable diabetic retinopathy detected. Prompt ophthalmology referral recommended.";

    case "Proliferative DR"

        recommendation = ...
            "Proliferative diabetic retinopathy detected. Urgent ophthalmology referral recommended.";

    otherwise

        recommendation = ...
            "Clinical review recommended.";

end

fprintf('%s\n',recommendation);

%% 12. SAVE EXPLANATION

resultsFolder = fullfile(projectRoot,'results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

[~,baseName,~] = fileparts(fileName);

outputFile = fullfile( ...
    resultsFolder, ...
    [baseName '_GradCAM.png']);

saveas(gcf,outputFile);

fprintf('\nExplanation saved to:\n');
fprintf('%s\n',outputFile);

fprintf('\n========================================\n');
fprintf('EXPLAINABLE SCREENING COMPLETE\n');
fprintf('========================================\n');