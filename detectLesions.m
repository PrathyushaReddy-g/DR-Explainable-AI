%% LESION-LEVEL CLINICAL EVIDENCE
% Candidate detection of:
%   1. Exudate-like bright regions
%   2. Hemorrhage / microaneurysm-like dark regions
%
% Prototype image-processing module.
% Candidate regions are NOT clinical diagnoses.
%
% MATLAB R2026a

clear;
clc;
close all;

%% ============================================================
% 1. PROJECT PATH
% =============================================================

projectRoot = ...
    'C:\Users\LENOVO\Documents\MATLAB\DR_Explainable_AI_New';

%% ============================================================
% 2. SELECT FUNDUS IMAGE
% =============================================================

fprintf('\n========================================\n');
fprintf('LESION-LEVEL CLINICAL EVIDENCE\n');
fprintf('========================================\n');

[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg', ...
    'Fundus Images (*.png, *.jpg, *.jpeg)'}, ...
    'Select Fundus Image');

if isequal(fileName,0)
    fprintf('No image selected.\n');
    return;
end

imageFile = fullfile(filePath,fileName);

fprintf('\nSelected image: %s\n',fileName);

%% ============================================================
% 3. READ IMAGE
% =============================================================

originalImage = imread(imageFile);

if size(originalImage,3) == 1

    originalImage = cat(3, ...
        originalImage, ...
        originalImage, ...
        originalImage);

end

%% ============================================================
% 4. RESIZE IMAGE
% =============================================================

image = imresize(originalImage,[512 512]);

%% ============================================================
% 5. RGB CHANNELS
% =============================================================

R = im2double(image(:,:,1));
G = im2double(image(:,:,2));
B = im2double(image(:,:,3));

grayImage = rgb2gray(image);

%% ============================================================
% 6. ROBUST RETINAL FIELD-OF-VIEW DETECTION
% =============================================================

fprintf('\nDetecting retinal field of view...\n');

% Estimate dark background

backgroundThreshold = 0.08;

initialMask = grayImage > backgroundThreshold;

% Remove small objects

initialMask = bwareaopen(initialMask,500);

% Fill holes

initialMask = imfill(initialMask,'holes');

% Keep largest connected component

cc = bwconncomp(initialMask);

if cc.NumObjects > 0

    componentSizes = ...
        cellfun(@numel,cc.PixelIdxList);

    [~,largestIndex] = ...
        max(componentSizes);

    fovMask = false(size(initialMask));

    fovMask(cc.PixelIdxList{largestIndex}) = true;

else

    % Fallback

    [X,Y] = meshgrid(1:512,1:512);

    centerX = 256;
    centerY = 256;

    radius = 230;

    fovMask = ...
        ((X-centerX).^2 + ...
        (Y-centerY).^2) <= radius^2;

end

%% ============================================================
% 7. CLEAN FOV MASK
% =============================================================

fovMask = imclose( ...
    fovMask, ...
    strel('disk',10));

fovMask = imfill( ...
    fovMask,'holes');

%% ============================================================
% 8. SHRINK FOV MASK SLIGHTLY
% =============================================================
% This removes artificial detections close to the black image
% boundary.

analysisMask = imerode( ...
    fovMask, ...
    strel('disk',8));

%% ============================================================
% 9. REMOVE IMAGE BORDER
% =============================================================

borderMask = true(512,512);

borderWidth = 20;

borderMask( ...
    borderWidth+1:end-borderWidth, ...
    borderWidth+1:end-borderWidth) = false;

analysisMask(borderMask) = false;

%% ============================================================
% 10. EXUDATE-LIKE BRIGHT LESION DETECTION
% =============================================================

fprintf('Detecting bright lesion candidates...\n');

% Green channel is useful for retinal lesion contrast

greenEnhanced = adapthisteq(G);

%% 10.1 Local bright structure enhancement

brightBackground = imopen( ...
    greenEnhanced, ...
    strel('disk',12));

brightTophat = ...
    greenEnhanced - brightBackground;

%% 10.2 Threshold bright structures

brightValues = ...
    brightTophat(analysisMask);

brightThreshold = ...
    prctile(brightValues,99);

exudateMask = ...
    brightTophat > brightThreshold;

%% 10.3 Restrict to retina

exudateMask = ...
    exudateMask & analysisMask;

%% 10.4 Remove tiny noise

exudateMask = ...
    bwareaopen(exudateMask,8);

%% 10.5 Morphological cleanup

exudateMask = ...
    imclose( ...
        exudateMask, ...
        strel('disk',2));

%% ============================================================
% 11. REMOVE LARGE / IRREGULAR BRIGHT STRUCTURES
% =============================================================
% Large structures are more likely to be optic disc or
% illumination artifacts than small lesion candidates.

exudateCC = bwconncomp(exudateMask);

cleanExudateMask = false(size(exudateMask));

if exudateCC.NumObjects > 0

    for k = 1:exudateCC.NumObjects

        regionPixels = ...
            exudateCC.PixelIdxList{k};

        area = numel(regionPixels);

        if area >= 8 && area <= 800

            cleanExudateMask(regionPixels) = true;

        end

    end

end

exudateMask = cleanExudateMask;

%% ============================================================
% 12. REMOVE BRIGHT REGIONS VERY CLOSE TO OPTIC DISC
% =============================================================
% Detect the strongest large bright area as a probable optic
% disc region and suppress nearby candidate pixels.

brightForDisc = ...
    imgaussfilt(greenEnhanced,8);

discThreshold = ...
    prctile(brightForDisc(analysisMask),99.5);

discCandidate = ...
    brightForDisc > discThreshold;

discCandidate = ...
    discCandidate & analysisMask;

discCandidate = ...
    bwareaopen(discCandidate,100);

discCC = bwconncomp(discCandidate);

if discCC.NumObjects > 0

    discSizes = ...
        cellfun(@numel,discCC.PixelIdxList);

    [~,discIndex] = ...
        max(discSizes);

    discMask = false(size(discCandidate));

    discMask( ...
        discCC.PixelIdxList{discIndex}) = true;

    % Expand optic-disc exclusion region

    discMask = imdilate( ...
        discMask, ...
        strel('disk',25));

    exudateMask(discMask) = false;

end

%% ============================================================
% 13. FINAL EXUDATE CLEANUP
% =============================================================

exudateMask = ...
    bwareaopen(exudateMask,8);

exudateCC = ...
    bwconncomp(exudateMask);

numExudates = ...
    exudateCC.NumObjects;

%% ============================================================
% 14. DARK LESION DETECTION
% =============================================================

fprintf('Detecting dark lesion candidates...\n');

%% 14.1 Enhance green channel

darkInput = ...
    adapthisteq(G);

%% 14.2 Black top-hat enhancement
% Highlights small dark structures against brighter retina.

darkTophat = ...
    imbothat( ...
        darkInput, ...
        strel('disk',6));

%% 14.3 Threshold

darkValues = ...
    darkTophat(analysisMask);

darkThreshold = ...
    prctile(darkValues,99);

darkCandidate = ...
    darkTophat > darkThreshold;

darkCandidate = ...
    darkCandidate & analysisMask;

%% ============================================================
% 15. REMOVE SMALL NOISE
% =============================================================

darkCandidate = ...
    bwareaopen(darkCandidate,4);

%% ============================================================
% 16. CONNECTED COMPONENT FILTERING
% =============================================================

darkCC = ...
    bwconncomp(darkCandidate);

cleanDarkMask = ...
    false(size(darkCandidate));

if darkCC.NumObjects > 0

    for k = 1:darkCC.NumObjects

        pixels = ...
            darkCC.PixelIdxList{k};

        area = numel(pixels);

        if area >= 4 && area <= 250

            cleanDarkMask(pixels) = true;

        end

    end

end

darkLesionMask = cleanDarkMask;

%% ============================================================
% 17. REMOVE ELONGATED VESSEL-LIKE REGIONS
% =============================================================
% Candidate microaneurysm / hemorrhage regions should generally
% be more compact than long vessel segments.

darkCC = ...
    bwconncomp(darkLesionMask);

finalDarkMask = ...
    false(size(darkLesionMask));

if darkCC.NumObjects > 0

    stats = regionprops( ...
        darkLesionMask, ...
        'Area', ...
        'Eccentricity', ...
        'MajorAxisLength', ...
        'MinorAxisLength', ...
        'PixelIdxList');

    for k = 1:numel(stats)

        area = stats(k).Area;
        eccentricity = stats(k).Eccentricity;

        majorAxis = ...
            stats(k).MajorAxisLength;

        minorAxis = ...
            stats(k).MinorAxisLength;

        if minorAxis > 0

            aspectRatio = ...
                majorAxis / minorAxis;

        else

            aspectRatio = Inf;

        end

        % Keep compact candidate regions

        if area >= 4 && ...
           area <= 250 && ...
           eccentricity < 0.97 && ...
           aspectRatio < 8

            finalDarkMask( ...
                stats(k).PixelIdxList) = true;

        end

    end

end

darkLesionMask = finalDarkMask;

%% ============================================================
% 18. FINAL DARK COMPONENT COUNT
% =============================================================

darkCC = ...
    bwconncomp(darkLesionMask);

numDarkCandidates = ...
    darkCC.NumObjects;

%% ============================================================
% 19. CALCULATE RETINAL COVERAGE
% =============================================================

fovCoverage = ...
    100 * nnz(fovMask) / numel(fovMask);

analysisCoverage = ...
    100 * nnz(analysisMask) / numel(analysisMask);

%% ============================================================
% 20. DISPLAY RESULTS
% =============================================================

fprintf('\n========================================\n');
fprintf('LESION CANDIDATE RESULTS\n');
fprintf('========================================\n');

fprintf('Retinal FOV coverage       : %.2f%%\n', ...
    fovCoverage);

fprintf('Analysis region coverage   : %.2f%%\n', ...
    analysisCoverage);

fprintf('Possible bright lesions    : %d\n', ...
    numExudates);

fprintf('Possible dark lesions      : %d\n', ...
    numDarkCandidates);

%% ============================================================
% 21. CREATE CLEAN FIGURE
% =============================================================

fig = figure( ...
    'Name','Lesion-Level Clinical Evidence', ...
    'NumberTitle','off', ...
    'Color','white', ...
    'Toolbar','none', ...
    'MenuBar','none');

set(fig, ...
    'Position',[100 100 1200 750]);

tiledlayout(2,2, ...
    'Padding','compact', ...
    'TileSpacing','compact');

%% ============================================================
% 22. ORIGINAL IMAGE
% =============================================================

nexttile;

imshow(image);

title( ...
    'Original Fundus Image', ...
    'FontSize',15, ...
    'FontWeight','bold');

%% ============================================================
% 23. BRIGHT LESION MAP
% =============================================================

nexttile;

imshow(exudateMask);

title( ...
    sprintf('Possible Bright Lesions (%d)', ...
    numExudates), ...
    'FontSize',15, ...
    'FontWeight','bold');

%% ============================================================
% 24. DARK LESION MAP
% =============================================================

nexttile;

imshow(darkLesionMask);

title( ...
    sprintf('Possible Dark Lesions (%d)', ...
    numDarkCandidates), ...
    'FontSize',15, ...
    'FontWeight','bold');

%% ============================================================
% 25. ANNOTATED CLINICAL EVIDENCE
% =============================================================

nexttile;

imshow(image);

hold on;

%% Bright lesion boundaries

if numExudates > 0

    visboundaries( ...
        exudateMask, ...
        'Color','r', ...
        'LineWidth',1.5);

end

%% Dark lesion boundaries

if numDarkCandidates > 0

    visboundaries( ...
        darkLesionMask, ...
        'Color','b', ...
        'LineWidth',1.2);

end

title( ...
    'Candidate Clinical Evidence', ...
    'FontSize',15, ...
    'FontWeight','bold');

hold off;

%% ============================================================
% 26. SAVE RESULT
% =============================================================

resultsFolder = ...
    fullfile(projectRoot,'results');

if ~exist(resultsFolder,'dir')

    mkdir(resultsFolder);

end

[~,baseName,~] = ...
    fileparts(fileName);

outputFile = fullfile( ...
    resultsFolder, ...
    [baseName '_LesionEvidence.png']);

exportgraphics( ...
    fig, ...
    outputFile, ...
    'Resolution',150);

fprintf('\nEvidence visualization saved to:\n');
fprintf('%s\n',outputFile);

%% ============================================================
% 27. CLINICAL EVIDENCE SUMMARY
% =============================================================

fprintf('\n========================================\n');
fprintf('EVIDENCE SUMMARY\n');
fprintf('========================================\n');

if numExudates > 0

    fprintf([ ...
        'Possible bright lesion candidates detected.\n' ...
        'These may correspond to exudate-like structures.\n']);

else

    fprintf( ...
        'No strong bright lesion candidates detected.\n');

end

if numDarkCandidates > 0

    fprintf([ ...
        'Possible dark lesion candidates detected.\n' ...
        'These may correspond to hemorrhage or ' ...
        'microaneurysm-like structures.\n']);

else

    fprintf( ...
        'No strong dark lesion candidates detected.\n');

end

fprintf('\n----------------------------------------\n');

fprintf([ ...
    'IMPORTANT: Candidate regions are generated by ' ...
    'image-processing methods and require clinical ' ...
    'validation. They are not diagnostic findings.\n']);

fprintf('========================================\n');
fprintf('LESION ANALYSIS COMPLETE\n');
fprintf('========================================\n');