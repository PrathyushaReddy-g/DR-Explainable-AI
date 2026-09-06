%% ============================================================
%  DIABETIC RETINOPATHY SCREENING
%  AI-Assisted Fundus Screening with Explainable AI
%
%  MATLAB R2026a
%
%  Pipeline:
%       Fundus Image
%            |
%       Image Quality Assessment
%            |
%       Quality Decision
%            |
%       ResNet DR Classification
%            |
%       Referable DR Decision
%            |
%       Grad-CAM Explainability
%            |
%       Professional Screening Report
%
%  NOTE:
%  This is an AI-assisted screening prototype.
%  It is not a definitive medical diagnosis.
%% ============================================================

clear;
clc;
close all;

%% ============================================================
% 1. PROJECT CONFIGURATION
%% ============================================================

projectRoot = ...
    'C:\Users\LENOVO\Documents\MATLAB\DR_Explainable_AI_New';

modelFile = fullfile( ...
    projectRoot, ...
    'models', ...
    'DR_Model.mat');

resultsFolder = fullfile( ...
    projectRoot, ...
    'results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

fprintf('\n');
fprintf('============================================================\n');
fprintf('        AI-ASSISTED DIABETIC RETINOPATHY SCREENING\n');
fprintf('============================================================\n');

%% ============================================================
% 2. LOAD TRAINED MODEL
%% ============================================================

fprintf('\nLoading trained model...\n');

if ~isfile(modelFile)
    error(['Model file not found:\n%s\n\n' ...
        'Please make sure DR_Model.mat exists inside the models folder.'], ...
        modelFile);
end

modelData = load(modelFile);

if ~isfield(modelData,'trainedNet')
    error('trainedNet was not found inside DR_Model.mat.');
end

trainedNet = modelData.trainedNet;

if isfield(modelData,'classNames')
    classNames = string(modelData.classNames);
else
    classNames = [ ...
        "No DR", ...
        "Mild DR", ...
        "Moderate DR", ...
        "Severe DR", ...
        "Proliferative DR"];
end

fprintf('Model loaded successfully.\n');

%% ============================================================
% 3. DEFINE DR CLASSES
%% ============================================================

classNames = string(classNames(:)');

numClasses = numel(classNames);

if numClasses ~= 5
    warning('Expected 5 DR classes, but found %d.',numClasses);
end

%% ============================================================
% 4. SELECT FUNDUS IMAGE
%% ============================================================

fprintf('\nSelect a fundus image for screening.\n');

[fileName,filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.tif;*.tiff', ...
    'Fundus Images (*.png, *.jpg, *.jpeg, *.tif, *.tiff)'}, ...
    'Select Fundus Image');

if isequal(fileName,0)
    fprintf('Screening cancelled.\n');
    return;
end

imageFile = fullfile(filePath,fileName);

fprintf('\nSelected image:\n%s\n',imageFile);

%% ============================================================
% 5. READ IMAGE
%% ============================================================

originalImage = imread(imageFile);

% Convert grayscale to RGB
if ndims(originalImage) == 2
    originalImage = cat(3, ...
        originalImage, ...
        originalImage, ...
        originalImage);
end

% Remove alpha channel
if size(originalImage,3) > 3
    originalImage = originalImage(:,:,1:3);
end

% Keep RGB uint8 image for display
if ~isa(originalImage,'uint8')
    originalImage = im2uint8(originalImage);
end

%% ============================================================
% 6. IMAGE QUALITY ASSESSMENT
%% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 IMAGE QUALITY ASSESSMENT\n');
fprintf('============================================================\n');

grayImage = rgb2gray(originalImage);
grayDouble = im2double(grayImage);

%% ------------------------------------------------------------
% 6.1 Retinal Field of View
%% ------------------------------------------------------------

maxGray = max(grayDouble(:));

if maxGray <= 0
    fovMask = false(size(grayDouble));
else
    fovMask = grayDouble > maxGray * 0.05;
end

fovMask = bwareaopen(fovMask,100);

if any(fovMask(:))

    ccFOV = bwconncomp(fovMask);

    regionSizes = cellfun( ...
        @numel, ...
        ccFOV.PixelIdxList);

    [~,largestIndex] = max(regionSizes);

    cleanFOV = false(size(fovMask));

    cleanFOV(ccFOV.PixelIdxList{largestIndex}) = true;

    fovMask = cleanFOV;

else

    fovMask = true(size(grayDouble));

end

fovScore = nnz(fovMask) / numel(fovMask);

%% ------------------------------------------------------------
% 6.2 Illumination
%% ------------------------------------------------------------

if any(fovMask(:))
    illuminationRaw = mean(grayDouble(fovMask));
else
    illuminationRaw = mean(grayDouble(:));
end

% Convert to a quality score.
% Good illumination is approximately in the middle range.

illuminationScore = ...
    1 - min(abs(illuminationRaw - 0.45) / 0.45,1);

illuminationScore = max(0,min(1,illuminationScore));

%% ------------------------------------------------------------
% 6.3 Contrast
%% ------------------------------------------------------------

if any(fovMask(:))
    retinalPixels = grayDouble(fovMask);
else
    retinalPixels = grayDouble(:);
end

contrastRaw = std(retinalPixels);

contrastScore = ...
    min(contrastRaw / 0.20,1);

contrastScore = max(0,min(1,contrastScore));

%% ------------------------------------------------------------
% 6.4 Focus / Sharpness
%% ------------------------------------------------------------

laplacianKernel = [ ...
     0 -1  0;
    -1  4 -1;
     0 -1  0];

laplacianImage = imfilter( ...
    grayDouble, ...
    laplacianKernel, ...
    'replicate');

% Do not use var() because it can be shadowed.
lapMean = mean(laplacianImage(:));

focusRaw = mean( ...
    (laplacianImage(:) - lapMean).^2);

% Normalize prototype focus score.
focusScore = min(focusRaw / 0.003,1);

focusScore = max(0,min(1,focusScore));

%% ============================================================
% 7. QUALITY DECISION
%% ============================================================

focusPercent = focusScore * 100;
illuminationPercent = illuminationScore * 100;
fovPercent = fovScore * 100;
contrastPercent = contrastScore * 100;

% Quality thresholds
poorFocus = focusPercent < 20;
poorIllumination = illuminationPercent < 25;
poorFOV = fovPercent < 50;
poorContrast = contrastPercent < 20;

% Borderline thresholds
borderlineFocus = ...
    focusPercent >= 20 && focusPercent < 40;

borderlineIllumination = ...
    illuminationPercent >= 25 && illuminationPercent < 45;

borderlineFOV = ...
    fovPercent >= 50 && fovPercent < 65;

borderlineContrast = ...
    contrastPercent >= 20 && contrastPercent < 35;

isPoor = ...
    poorFocus || ...
    poorIllumination || ...
    poorFOV || ...
    poorContrast;

isBorderline = ...
    ~isPoor && ...
    (borderlineFocus || ...
     borderlineIllumination || ...
     borderlineFOV || ...
     borderlineContrast);

%% ------------------------------------------------------------
% Overall Quality Score
%%

qualityScore = ...
    0.30 * focusScore + ...
    0.25 * illuminationScore + ...
    0.25 * fovScore + ...
    0.20 * contrastScore;

qualityScore = max(0,min(1,qualityScore));

overallQualityPercent = qualityScore * 100;

if isPoor

    qualityStatus = "RECAPTURE REQUIRED";

elseif isBorderline

    qualityStatus = "BORDERLINE";

else

    qualityStatus = "ACCEPTABLE";

end

%% ============================================================
% 8. QUALITY FEEDBACK
%% ============================================================

feedback = strings(0);

if poorFocus
    feedback(end+1) = ...
        "Image appears blurred or insufficiently focused.";
end

if poorIllumination
    feedback(end+1) = ...
        "Image illumination is inadequate or uneven.";
end

if poorFOV
    feedback(end+1) = ...
        "Retinal field of view is insufficient.";
end

if poorContrast
    feedback(end+1) = ...
        "Image contrast is low; retinal structures may be difficult to assess.";
end

if isempty(feedback)

    if isBorderline
        feedback(end+1) = ...
            "Image quality is borderline; enhancement will be applied.";
    else
        feedback(end+1) = ...
            "Image quality is acceptable for downstream screening.";
    end

end

fprintf('\nFocus / Sharpness : %.1f%%\n',focusPercent);
fprintf('Illumination       : %.1f%%\n',illuminationPercent);
fprintf('FOV Coverage       : %.1f%%\n',fovPercent);
fprintf('Contrast           : %.1f%%\n',contrastPercent);
fprintf('Overall Quality    : %.1f%%\n',overallQualityPercent);
fprintf('Quality Decision   : %s\n',qualityStatus);

%% ============================================================
% 9. ENHANCE IMAGE IF BORDERLINE
%% ============================================================

processedImage = originalImage;

if isBorderline

    fprintf('\nApplying image enhancement...\n');

    labImage = rgb2lab(im2double(originalImage));

    Lchannel = labImage(:,:,1) / 100;

    Lchannel = adapthisteq( ...
        Lchannel, ...
        'NumTiles',[8 8], ...
        'ClipLimit',0.01);

    labImage(:,:,1) = Lchannel * 100;

    processedImage = lab2rgb(labImage);

    processedImage = ...
        im2uint8(processedImage);

else

    fprintf('\nNo additional enhancement required.\n');

end

%% ============================================================
% 10. STOP IF IMAGE IS TOO POOR
%% ============================================================

if isPoor

    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('SCREENING STOPPED\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('The image did not meet minimum quality requirements.\n');
    fprintf('Please recapture the fundus image before DR analysis.\n');
    fprintf('------------------------------------------------------------\n');

    createQualityOnlyReport( ...
        originalImage, ...
        fovMask, ...
        processedImage, ...
        fileName, ...
        qualityStatus, ...
        overallQualityPercent, ...
        focusPercent, ...
        illuminationPercent, ...
        fovPercent, ...
        contrastPercent, ...
        feedback, ...
        resultsFolder);

    return;

end

%% ============================================================
% 11. PREPARE IMAGE FOR NETWORK
%% ============================================================

inputSize = [224 224];

networkImage = imresize( ...
    processedImage, ...
    inputSize);

networkImage = im2single(networkImage);

%% ============================================================
% 12. DR CLASSIFICATION
%% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                    DR CLASSIFICATION\n');
fprintf('============================================================\n');

scores = predict( ...
    trainedNet, ...
    networkImage);

if isa(scores,'dlarray')
    scores = extractdata(scores);
end

scores = gather(scores);

scores = double(scores);

scores = squeeze(scores);

scores = scores(:);

%% ------------------------------------------------------------
% Convert model output to probabilities
%% ------------------------------------------------------------

if all(isfinite(scores))

    if all(scores >= 0) && ...
            abs(sum(scores) - 1) < 1e-3

        probabilities = scores;

    else

        shiftedScores = ...
            scores - max(scores);

        probabilities = ...
            exp(shiftedScores);

        probabilitySum = sum(probabilities);

        if probabilitySum > 0
            probabilities = ...
                probabilities / probabilitySum;
        else
            probabilities = ...
                ones(size(scores)) / numel(scores);
        end

    end

else

    error('The trained model returned invalid prediction scores.');

end

%% ------------------------------------------------------------
% Match number of classes
%%

if numel(probabilities) ~= numClasses

    error(['Model returned %d prediction scores, but %d ' ...
        'class names were found.'], ...
        numel(probabilities),numClasses);

end

%% ============================================================
% 13. PREDICTED CLASS
%% ============================================================

[confidence,classIndex] = max(probabilities);

predictedClass = classNames(classIndex);

confidencePercent = confidence * 100;

fprintf('\nPredicted DR Grade : %d\n',classIndex-1);
fprintf('Predicted Class    : %s\n',predictedClass);
fprintf('Confidence         : %.1f%%\n',confidencePercent);

%% ============================================================
% 14. REFERABLE DR DECISION
%
% International Clinical DR Scale:
%
% Grade 0 = No DR
% Grade 1 = Mild DR
% Grade 2 = Moderate DR
% Grade 3 = Severe DR
% Grade 4 = Proliferative DR
%
% Referable DR = Grade 2 or higher
%% ============================================================

if classIndex >= 3

    isReferable = true;

    referableStatus = "REFERABLE DR";

else

    isReferable = false;

    referableStatus = "NON-REFERABLE DR";

end

%% ============================================================
% 15. CLINICAL SCREENING RECOMMENDATION
%% ============================================================

switch classIndex

    case 1

        recommendationTitle = ...
            "Routine screening follow-up";

        recommendationText = ...
            "No diabetic retinopathy detected by the AI model.";

    case 2

        recommendationTitle = ...
            "Clinical follow-up recommended";

        recommendationText = ...
            "Mild diabetic retinopathy pattern detected. Clinical follow-up is recommended.";

    case 3

        recommendationTitle = ...
            "Ophthalmology referral recommended";

        recommendationText = ...
            "Moderate diabetic retinopathy pattern detected. Ophthalmic evaluation is recommended.";

    case 4

        recommendationTitle = ...
            "Prompt ophthalmology referral";

        recommendationText = ...
            "Severe diabetic retinopathy pattern detected. Prompt ophthalmic evaluation is recommended.";

    case 5

        recommendationTitle = ...
            "Urgent ophthalmology evaluation";

        recommendationText = ...
            "Proliferative diabetic retinopathy pattern detected. Urgent ophthalmic evaluation is recommended.";

    otherwise

        recommendationTitle = ...
            "Clinical review recommended";

        recommendationText = ...
            "AI screening result requires review by a qualified eye-care professional.";

end

%% ============================================================
% 16. DISPLAY PROBABILITIES IN COMMAND WINDOW
%% ============================================================

fprintf('\nModel probability distribution:\n');

for k = 1:numClasses

    fprintf('  %-22s %6.2f%%\n', ...
        classNames(k), ...
        probabilities(k)*100);

end

%% ============================================================
% 17. GRAD-CAM EXPLAINABILITY
%% ============================================================

fprintf('\nGenerating Grad-CAM explanation...\n');

try

    scoreMap = gradCAM( ...
        trainedNet, ...
        networkImage, ...
        classIndex);

    scoreMap = gather(scoreMap);

    scoreMap = squeeze(scoreMap);

    scoreMap = mat2gray(scoreMap);

    scoreMap = imresize( ...
        scoreMap, ...
        [size(originalImage,1), ...
         size(originalImage,2)]);

    gradCAMStatus = "Generated";

    fprintf('Grad-CAM generated successfully.\n');

catch ME

    scoreMap = [];

    gradCAMStatus = "Unavailable";

    fprintf('\nGrad-CAM could not be generated.\n');
    fprintf('Reason: %s\n',ME.message);

end

%% ============================================================
% 18. CREATE PROFESSIONAL SCREENING REPORT
%% ============================================================

fprintf('\nCreating professional screening report...\n');

fig = figure( ...
    'Name','AI-Assisted Diabetic Retinopathy Screening Report', ...
    'NumberTitle','off', ...
    'Color',[0.94 0.95 0.97], ...
    'Units','normalized', ...
    'Position',[0.03 0.05 0.94 0.88]);

%% ------------------------------------------------------------
% Header
%%

annotation(fig,'rectangle', ...
    [0 0.925 1 0.075], ...
    'FaceColor',[0.035 0.13 0.22], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.04 0.947 0.65 0.04], ...
    'String','DIABETIC RETINOPATHY SCREENING', ...
    'Color','white', ...
    'FontSize',24, ...
    'FontWeight','bold', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','left');

annotation(fig,'textbox', ...
    [0.04 0.925 0.60 0.025], ...
    'String','AI-ASSISTED FUNDUS SCREENING REPORT', ...
    'Color',[0.75 0.84 0.92], ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','left');

dateText = datestr(now,'dd mmm yyyy | HH:MM');

annotation(fig,'textbox', ...
    [0.75 0.947 0.21 0.035], ...
    'String',dateText, ...
    'Color','white', ...
    'FontSize',11, ...
    'EdgeColor','none', ...
    'HorizontalAlignment','right');

%% ============================================================
% 19. IMAGE PANEL
%% ============================================================

axImage = axes( ...
    'Parent',fig, ...
    'Position',[0.07 0.53 0.30 0.32]);

imshow(originalImage,'Parent',axImage);

title(axImage, ...
    'FUNDUS IMAGE', ...
    'FontSize',18, ...
    'FontWeight','bold');

%% ============================================================
% 20. SCREENING RESULT PANEL
%%

annotation(fig,'rectangle', ...
    [0.43 0.53 0.51 0.32], ...
    'FaceColor','white', ...
    'EdgeColor',[0.86 0.88 0.91], ...
    'LineWidth',1.2);

annotation(fig,'textbox', ...
    [0.46 0.80 0.43 0.035], ...
    'String','SCREENING RESULT', ...
    'FontSize',17, ...
    'FontWeight','bold', ...
    'Color',[0.10 0.12 0.15], ...
    'EdgeColor','none');

gradeText = sprintf('GRADE %d',classIndex-1);

annotation(fig,'textbox', ...
    [0.46 0.735 0.42 0.065], ...
    'String',gradeText, ...
    'FontSize',32, ...
    'FontWeight','bold', ...
    'Color',[0.08 0.12 0.16], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.46 0.695 0.42 0.045], ...
    'String',char(upper(predictedClass)), ...
    'FontSize',18, ...
    'FontWeight','bold', ...
    'Color',[0.08 0.12 0.16], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.46 0.645 0.25 0.035], ...
    'String','Model confidence', ...
    'FontSize',13, ...
    'Color',[0.30 0.33 0.37], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.77 0.645 0.12 0.035], ...
    'String',sprintf('%.1f%%',confidencePercent), ...
    'FontSize',15, ...
    'FontWeight','bold', ...
    'Color',[0.08 0.12 0.16], ...
    'EdgeColor','none', ...
    'HorizontalAlignment','right');

% Confidence background
annotation(fig,'rectangle', ...
    [0.46 0.625 0.42 0.012], ...
    'FaceColor',[0.88 0.90 0.93], ...
    'EdgeColor','none');

% Confidence foreground
annotation(fig,'rectangle', ...
    [0.46 0.625 0.42*confidence 0.012], ...
    'FaceColor',[0.09 0.48 0.73], ...
    'EdgeColor','none');

%% Referable status

if isReferable

    statusColor = [0.72 0.16 0.10];

else

    statusColor = [0.08 0.48 0.25];

end

annotation(fig,'textbox', ...
    [0.46 0.55 0.43 0.045], ...
    'String',char(referableStatus), ...
    'FontSize',18, ...
    'FontWeight','bold', ...
    'Color',statusColor, ...
    'EdgeColor','none');

%% ============================================================
% 21. PROBABILITY DISTRIBUTION
%%

annotation(fig,'rectangle', ...
    [0.05 0.12 0.43 0.34], ...
    'FaceColor','white', ...
    'EdgeColor',[0.86 0.88 0.91], ...
    'LineWidth',1.2);

annotation(fig,'textbox', ...
    [0.08 0.415 0.36 0.035], ...
    'String','MODEL PROBABILITY DISTRIBUTION', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'Color',[0.10 0.12 0.15], ...
    'EdgeColor','none');

%% Probability axes

axProb = axes( ...
    'Parent',fig, ...
    'Position',[0.09 0.16 0.35 0.23]);

barh(axProb, ...
    1:numClasses, ...
    probabilities*100);

axProb.YDir = 'reverse';

axProb.YTick = 1:numClasses;
axProb.YTickLabel = cellstr(classNames);

axProb.XLim = [0 100];

axProb.XTick = [0 25 50 75 100];

axProb.XGrid = 'on';

axProb.GridAlpha = 0.15;

axProb.FontSize = 10;

axProb.Box = 'off';

xlabel(axProb,'Probability (%)');

%% ============================================================
% 22. QUALITY SUMMARY PANEL
%%

annotation(fig,'rectangle', ...
    [0.51 0.12 0.43 0.34], ...
    'FaceColor','white', ...
    'EdgeColor',[0.86 0.88 0.91], ...
    'LineWidth',1.2);

annotation(fig,'textbox', ...
    [0.54 0.415 0.36 0.035], ...
    'String','IMAGE QUALITY ASSESSMENT', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'Color',[0.10 0.12 0.15], ...
    'EdgeColor','none');

%% Quality decision

if isPoor

    qualityColor = [0.72 0.16 0.10];

elseif isBorderline

    qualityColor = [0.70 0.43 0.05];

else

    qualityColor = [0.08 0.48 0.25];

end

annotation(fig,'textbox', ...
    [0.54 0.365 0.36 0.04], ...
    'String',char(qualityStatus), ...
    'FontSize',20, ...
    'FontWeight','bold', ...
    'Color',qualityColor, ...
    'EdgeColor','none');

%% Quality metrics

metricNames = { ...
    'Focus / Sharpness', ...
    'Illumination', ...
    'FOV Coverage', ...
    'Contrast'};

metricValues = [ ...
    focusPercent, ...
    illuminationPercent, ...
    fovPercent, ...
    contrastPercent];

metricY = [0.315 0.270 0.225 0.180];

for k = 1:4

    annotation(fig,'textbox', ...
        [0.54 metricY(k) 0.16 0.025], ...
        'String',metricNames{k}, ...
        'FontSize',11, ...
        'FontWeight','bold', ...
        'Color',[0.25 0.28 0.32], ...
        'EdgeColor','none');

    % Background bar

    annotation(fig,'rectangle', ...
        [0.69 metricY(k)+0.006 0.17 0.012], ...
        'FaceColor',[0.88 0.90 0.93], ...
        'EdgeColor','none');

    % Value bar

    annotation(fig,'rectangle', ...
        [0.69 metricY(k)+0.006 ...
         0.17*metricValues(k)/100 ...
         0.012], ...
        'FaceColor',[0.09 0.48 0.73], ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.87 metricY(k)-0.002 0.05 0.025], ...
        'String',sprintf('%.0f%%',metricValues(k)), ...
        'FontSize',10, ...
        'FontWeight','bold', ...
        'Color',[0.20 0.23 0.27], ...
        'EdgeColor','none', ...
        'HorizontalAlignment','right');

end

%% ============================================================
% 23. CLINICAL RECOMMENDATION
%%

annotation(fig,'textbox', ...
    [0.54 0.135 0.36 0.035], ...
    'String','CLINICAL SCREENING RECOMMENDATION', ...
    'FontSize',14, ...
    'FontWeight','bold', ...
    'Color',[0.10 0.12 0.15], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.54 0.095 0.37 0.035], ...
    'String',char(recommendationTitle), ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'Color',[0.10 0.12 0.15], ...
    'EdgeColor','none');

annotation(fig,'textbox', ...
    [0.54 0.045 0.37 0.055], ...
    'String',char(recommendationText), ...
    'FontSize',10, ...
    'Color',[0.32 0.35 0.39], ...
    'EdgeColor','none', ...
    'VerticalAlignment','top');

%% ============================================================
% 24. GRAD-CAM EXPLANATION WINDOW
%%

if ~isempty(scoreMap)

    gradFig = figure( ...
        'Name','DR Explainability - Grad-CAM', ...
        'NumberTitle','off', ...
        'Color','white', ...
        'Units','normalized', ...
        'Position',[0.12 0.12 0.76 0.72]);

    tl = tiledlayout(1,3, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    title(tl, ...
        sprintf('EXPLAINABLE DR SCREENING | GRADE %d | %s', ...
        classIndex-1, ...
        predictedClass), ...
        'FontSize',18, ...
        'FontWeight','bold');

    % Original

    ax1 = nexttile;

    imshow(originalImage,'Parent',ax1);

    title(ax1, ...
        'Original Fundus', ...
        'FontSize',14, ...
        'FontWeight','bold');

    % Grad-CAM

    ax2 = nexttile;

    imagesc(ax2,scoreMap);

    axis(ax2,'image');
    axis(ax2,'off');

    title(ax2, ...
        'Grad-CAM Attention', ...
        'FontSize',14, ...
        'FontWeight','bold');

    colormap(ax2,jet);

    colorbar(ax2);

    % Overlay

    ax3 = nexttile;

    imshow(originalImage,'Parent',ax3);

    hold(ax3,'on');

    imagesc(ax3,scoreMap, ...
        'AlphaData',0.45);

    axis(ax3,'image');
    axis(ax3,'off');

    title(ax3, ...
        'Clinical Explanation Overlay', ...
        'FontSize',14, ...
        'FontWeight','bold');

    hold(ax3,'off');

end

%% ============================================================
% 25. SAVE SCREENING REPORT
%% ============================================================

[~,baseName,~] = fileparts(fileName);

reportFile = fullfile( ...
    resultsFolder, ...
    [baseName '_ProfessionalScreeningReport.png']);

saveas(fig,reportFile);

%% ============================================================
% 26. SAVE SCREENING DATA
%%

resultFile = fullfile( ...
    resultsFolder, ...
    [baseName '_ScreeningResult.mat']);

screeningResult = struct();

screeningResult.imageFile = imageFile;

screeningResult.fileName = fileName;

screeningResult.qualityStatus = qualityStatus;

screeningResult.qualityScore = overallQualityPercent;

screeningResult.focusScore = focusPercent;

screeningResult.illuminationScore = illuminationPercent;

screeningResult.fovScore = fovPercent;

screeningResult.contrastScore = contrastPercent;

screeningResult.predictedGrade = classIndex - 1;

screeningResult.predictedClass = predictedClass;

screeningResult.confidence = confidencePercent;

screeningResult.classProbabilities = probabilities;

screeningResult.classNames = classNames;

screeningResult.isReferable = isReferable;

screeningResult.referableStatus = referableStatus;

screeningResult.recommendationTitle = recommendationTitle;

screeningResult.recommendationText = recommendationText;

screeningResult.gradCAMStatus = gradCAMStatus;

screeningResult.feedback = feedback;

screeningResult.timestamp = datetime('now');

save(resultFile,'screeningResult');

%% ============================================================
% 27. FINAL COMMAND WINDOW SUMMARY
%%

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 FINAL SCREENING SUMMARY\n');
fprintf('============================================================\n');

fprintf('Image               : %s\n',fileName);

fprintf('Image Quality       : %s\n',qualityStatus);

fprintf('Quality Score       : %.1f%%\n',overallQualityPercent);

fprintf('DR Grade            : %d\n',classIndex-1);

fprintf('DR Classification   : %s\n',predictedClass);

fprintf('Model Confidence    : %.1f%%\n',confidencePercent);

fprintf('Referable DR        : %s\n',referableStatus);

fprintf('Grad-CAM            : %s\n',gradCAMStatus);

fprintf('\nRecommendation:\n');
fprintf('%s\n',recommendationText);

fprintf('\nProfessional report saved to:\n');
fprintf('%s\n',reportFile);

fprintf('\nScreening data saved to:\n');
fprintf('%s\n',resultFile);

fprintf('\n============================================================\n');
fprintf('              SCREENING PROCESS COMPLETED\n');
fprintf('============================================================\n');

fprintf('\nIMPORTANT:\n');
fprintf(['This AI-assisted result is intended for screening support ' ...
    'and requires clinical review by a qualified eye-care professional.\n']);

%% ============================================================
% LOCAL FUNCTION
% QUALITY-ONLY REPORT FOR POOR IMAGES
%% ============================================================

function createQualityOnlyReport( ...
    originalImage, ...
    fovMask, ...
    processedImage, ...
    fileName, ...
    qualityStatus, ...
    overallQualityPercent, ...
    focusPercent, ...
    illuminationPercent, ...
    fovPercent, ...
    contrastPercent, ...
    feedback, ...
    resultsFolder)

    fig = figure( ...
        'Name','Fundus Image Quality Assessment', ...
        'NumberTitle','off', ...
        'Color',[0.94 0.95 0.97], ...
        'Units','normalized', ...
        'Position',[0.05 0.08 0.90 0.82]);

    %% Header

    annotation(fig,'rectangle', ...
        [0 0.91 1 0.09], ...
        'FaceColor',[0.035 0.13 0.22], ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.04 0.945 0.75 0.04], ...
        'String','FUNDUS IMAGE QUALITY ASSESSMENT', ...
        'Color','white', ...
        'FontSize',23, ...
        'FontWeight','bold', ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.04 0.915 0.70 0.025], ...
        'String','AI-ASSISTED DIABETIC RETINOPATHY SCREENING', ...
        'Color',[0.75 0.84 0.92], ...
        'FontSize',11, ...
        'FontWeight','bold', ...
        'EdgeColor','none');

    %% Original image

    ax1 = axes( ...
        'Parent',fig, ...
        'Position',[0.07 0.49 0.25 0.34]);

    imshow(originalImage,'Parent',ax1);

    title(ax1, ...
        'ORIGINAL FUNDUS IMAGE', ...
        'FontSize',15, ...
        'FontWeight','bold');

    %% FOV

    ax2 = axes( ...
        'Parent',fig, ...
        'Position',[0.375 0.49 0.25 0.34]);

    imshow(fovMask,'Parent',ax2);

    title(ax2, ...
        sprintf('RETINAL FIELD OF VIEW | %.1f%%',fovPercent), ...
        'FontSize',15, ...
        'FontWeight','bold');

    %% Enhanced

    ax3 = axes( ...
        'Parent',fig, ...
        'Position',[0.68 0.49 0.25 0.34]);

    imshow(processedImage,'Parent',ax3);

    title(ax3, ...
        'ENHANCED RETINAL VIEW', ...
        'FontSize',15, ...
        'FontWeight','bold');

    %% Bottom panel

    annotation(fig,'rectangle', ...
        [0.05 0.08 0.90 0.34], ...
        'FaceColor','white', ...
        'EdgeColor',[0.86 0.88 0.91], ...
        'LineWidth',1.2);

    annotation(fig,'textbox', ...
        [0.08 0.36 0.40 0.035], ...
        'String','QUALITY DECISION', ...
        'FontSize',16, ...
        'FontWeight','bold', ...
        'Color',[0.10 0.12 0.15], ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.08 0.305 0.45 0.055], ...
        'String','RECAPTURE REQUIRED', ...
        'FontSize',25, ...
        'FontWeight','bold', ...
        'Color',[0.72 0.16 0.10], ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.65 0.36 0.25 0.035], ...
        'String','OVERALL QUALITY SCORE', ...
        'FontSize',15, ...
        'FontWeight','bold', ...
        'Color',[0.10 0.12 0.15], ...
        'EdgeColor','none');

    annotation(fig,'textbox', ...
        [0.65 0.30 0.25 0.06], ...
        'String',sprintf('%.1f%%',overallQualityPercent), ...
        'FontSize',30, ...
        'FontWeight','bold', ...
        'Color',[0.05 0.15 0.25], ...
        'EdgeColor','none');

    %% Metric labels

    metricNames = { ...
        'Focus / Sharpness', ...
        'Illumination', ...
        'FOV Coverage', ...
        'Contrast'};

    metricValues = [ ...
        focusPercent, ...
        illuminationPercent, ...
        fovPercent, ...
        contrastPercent];

    yPositions = [0.235 0.195 0.155 0.115];

    for k = 1:4

        annotation(fig,'textbox', ...
            [0.08 yPositions(k) 0.18 0.025], ...
            'String',metricNames{k}, ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'Color',[0.25 0.28 0.32], ...
            'EdgeColor','none');

        annotation(fig,'rectangle', ...
            [0.25 yPositions(k)+0.006 0.27 0.012], ...
            'FaceColor',[0.88 0.90 0.93], ...
            'EdgeColor','none');

        annotation(fig,'rectangle', ...
            [0.25 yPositions(k)+0.006 ...
             0.27*metricValues(k)/100 ...
             0.012], ...
            'FaceColor',[0.09 0.48 0.73], ...
            'EdgeColor','none');

        annotation(fig,'textbox', ...
            [0.53 yPositions(k)-0.002 0.05 0.025], ...
            'String',sprintf('%.0f%%',metricValues(k)), ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'EdgeColor','none');

    end

    %% Feedback

    annotation(fig,'textbox', ...
        [0.65 0.23 0.25 0.035], ...
        'String','IMAGE QUALITY FEEDBACK', ...
        'FontSize',14, ...
        'FontWeight','bold', ...
        'Color',[0.10 0.12 0.15], ...
        'EdgeColor','none');

    feedbackText = '';

    for k = 1:numel(feedback)

        feedbackText = ...
            [feedbackText char(feedback(k)) newline newline]; %#ok<AGROW>

    end

    annotation(fig,'textbox', ...
        [0.65 0.10 0.27 0.12], ...
        'String',feedbackText, ...
        'FontSize',10, ...
        'Color',[0.30 0.33 0.37], ...
        'EdgeColor','none', ...
        'VerticalAlignment','top');

    %% Save

    [~,baseName,~] = fileparts(fileName);

    outputFile = fullfile( ...
        resultsFolder, ...
        [baseName '_QualityAssessment.png']);

    saveas(fig,outputFile);

    fprintf('\nQuality assessment report saved to:\n');
    fprintf('%s\n',outputFile);

end