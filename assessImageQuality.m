%% IMAGE QUALITY ASSESSMENT FOR DR SCREENING
% Checks:
% 1. Focus / sharpness
% 2. Illumination
% 3. Field of View (FOV)
%
% MATLAB R2026a

clear;
clc;
close all;

%% 1. PROJECT PATH

projectRoot = 'C:\Users\LENOVO\Documents\MATLAB\DR_Explainable_AI_New';

%% 2. SELECT FUNDUS IMAGE

fprintf('\n========================================\n');
fprintf('FUNDUS IMAGE QUALITY ASSESSMENT\n');
fprintf('========================================\n');

[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg','Fundus Images (*.png, *.jpg, *.jpeg)'}, ...
    'Select Fundus Image');

if isequal(fileName,0)
    fprintf('No image selected.\n');
    return;
end

imageFile = fullfile(filePath,fileName);

fprintf('\nSelected image: %s\n',fileName);

%% 3. READ IMAGE

originalImage = imread(imageFile);

%% 4. CONVERT TO RGB IF NECESSARY

if size(originalImage,3) == 1
    originalImage = cat(3, ...
        originalImage, ...
        originalImage, ...
        originalImage);
end

%% 5. RESIZE FOR ANALYSIS

analysisImage = imresize(originalImage,[512 512]);

grayImage = im2gray(analysisImage);
grayImage = im2double(grayImage);

%% 6. CREATE FUNDUS FIELD-OF-VIEW MASK

% Detect non-black retinal region

fovMask = grayImage > 0.08;

% Remove small regions
fovMask = bwareaopen(fovMask,500);

% Keep largest connected region
connected = bwconncomp(fovMask);

if connected.NumObjects > 0

    regionSizes = cellfun(@numel,connected.PixelIdxList);
    [~,largestIndex] = max(regionSizes);

    tempMask = false(size(fovMask));
    tempMask(connected.PixelIdxList{largestIndex}) = true;

    fovMask = imfill(tempMask,'holes');

end

%% 7. FIELD OF VIEW SCORE

totalPixels = numel(fovMask);
retinalPixels = nnz(fovMask);

fovPercentage = ...
    (retinalPixels / totalPixels) * 100;

%% 8. FOCUS / SHARPNESS ANALYSIS

% Laplacian-based focus metric

laplacianKernel = [ ...
     0 -1  0;
    -1  4 -1;
     0 -1  0];

laplacianImage = imfilter( ...
    grayImage, ...
    laplacianKernel, ...
    'replicate');

focusScore = var( ...
    laplacianImage(fovMask));

%% 9. ILLUMINATION ANALYSIS

retinalIntensity = grayImage(fovMask);

meanIntensity = mean(retinalIntensity);

illuminationStd = std(retinalIntensity);

%% 10. NORMALIZED QUALITY SCORES

% Focus score
if focusScore >= 0.003
    focusQuality = 100;
elseif focusScore >= 0.0015
    focusQuality = 75;
elseif focusScore >= 0.0007
    focusQuality = 50;
else
    focusQuality = 25;
end

% Illumination score

if meanIntensity >= 0.25 && meanIntensity <= 0.75
    illuminationQuality = 100;
elseif meanIntensity >= 0.15 && meanIntensity <= 0.85
    illuminationQuality = 75;
elseif meanIntensity >= 0.10 && meanIntensity <= 0.90
    illuminationQuality = 50;
else
    illuminationQuality = 25;
end

% FOV score

if fovPercentage >= 45
    fovQuality = 100;
elseif fovPercentage >= 30
    fovQuality = 75;
elseif fovPercentage >= 20
    fovQuality = 50;
else
    fovQuality = 25;
end

%% 11. OVERALL QUALITY SCORE

overallScore = mean([ ...
    focusQuality, ...
    illuminationQuality, ...
    fovQuality]);

%% 12. QUALITY DECISION

if overallScore >= 70

    qualityStatus = "ACCEPTED";

elseif overallScore >= 50

    qualityStatus = "BORDERLINE";

else

    qualityStatus = "REJECTED";

end

%% 13. RECATURE / QUALITY FEEDBACK

feedback = "";

if focusQuality < 70
    feedback = feedback + ...
        "Image is blurry. Please hold the camera steady and recapture. ";
end

if illuminationQuality < 70

    if meanIntensity < 0.25
        feedback = feedback + ...
            "Image is too dark. Improve illumination and recapture. ";
    else
        feedback = feedback + ...
            "Image is overexposed. Reduce illumination and recapture. ";
    end

end

if fovQuality < 70
    feedback = feedback + ...
        "Retinal field of view is insufficient. Center the retina and recapture. ";
end

if qualityStatus == "ACCEPTED"

    feedback = ...
        "Image quality is sufficient for DR analysis.";

elseif qualityStatus == "BORDERLINE"

    feedback = ...
        "Image quality is borderline. Consider recapturing for better reliability.";

end

%% 14. DISPLAY RESULTS IN COMMAND WINDOW

fprintf('\n========================================\n');
fprintf('IMAGE QUALITY RESULTS\n');
fprintf('========================================\n');

fprintf('Focus Score          : %.6f\n',focusScore);
fprintf('Focus Quality        : %.1f%%\n',focusQuality);

fprintf('Mean Illumination    : %.3f\n',meanIntensity);
fprintf('Illumination Quality : %.1f%%\n',illuminationQuality);

fprintf('FOV Coverage         : %.2f%%\n',fovPercentage);
fprintf('FOV Quality          : %.1f%%\n',fovQuality);

fprintf('----------------------------------------\n');

fprintf('Overall Quality      : %.1f%%\n',overallScore);
fprintf('Status               : %s\n',qualityStatus);

fprintf('----------------------------------------\n');

fprintf('Feedback:\n');
fprintf('%s\n',feedback);

fprintf('========================================\n');

%% 15. DISPLAY VISUAL QUALITY ANALYSIS

figure( ...
    'Name','Fundus Image Quality Assessment', ...
    'NumberTitle','off', ...
    'Color','white');

tiledlayout(1,3, ...
    'Padding','compact', ...
    'TileSpacing','compact');

%% ORIGINAL IMAGE

nexttile;

imshow(originalImage);

title( ...
    'Original Fundus Image', ...
    'FontSize',14, ...
    'FontWeight','bold');

%% FOV MASK

nexttile;

imshow(fovMask);

title( ...
    sprintf('Retinal Field of View\n%.1f%% Coverage',fovPercentage), ...
    'FontSize',14, ...
    'FontWeight','bold');

%% QUALITY RESULT

nexttile;

imshow(originalImage);

hold on;

visboundaries( ...
    fovMask, ...
    'Color','g', ...
    'LineWidth',1);

hold off;

title( ...
    sprintf('Quality: %s\nScore: %.1f%%', ...
    qualityStatus,overallScore), ...
    'FontSize',14, ...
    'FontWeight','bold');

%% 16. SAVE RESULT

resultsFolder = fullfile( ...
    projectRoot, ...
    'results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

[~,baseName,~] = fileparts(fileName);

outputFile = fullfile( ...
    resultsFolder, ...
    [baseName '_QualityAssessment.png']);

saveas(gcf,outputFile);

fprintf('\nQuality assessment saved to:\n');
fprintf('%s\n',outputFile);

fprintf('\n========================================\n');
fprintf('QUALITY ASSESSMENT COMPLETE\n');
fprintf('========================================\n');