%% ============================================================
% PROFESSIONAL DIABETIC RETINOPATHY SCREENING REPORT
% MATLAB R2026a
%
% International Clinical DR Scale
% 0 = No DR
% 1 = Mild DR
% 2 = Moderate DR
% 3 = Severe DR
% 4 = Proliferative DR
%
% Referable DR = Grade 2, 3 or 4
%
% AI-assisted screening prototype.
% Results require clinical review.
%% ============================================================

clear;
clc;
close all;

%% ============================================================
% 1. PROJECT PATH
% =============================================================

projectRoot = ...
    'C:\Users\LENOVO\Documents\MATLAB\DR_Explainable_AI_New';

modelFile = ...
    fullfile(projectRoot,'models','DR_Model.mat');

resultsFolder = ...
    fullfile(projectRoot,'results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

%% ============================================================
% 2. LOAD MODEL
% =============================================================

fprintf('\n========================================\n');
fprintf('DIABETIC RETINOPATHY SCREENING\n');
fprintf('========================================\n');

fprintf('\nLoading trained model...\n');

load(modelFile, ...
    'trainedNet', ...
    'classNames');

fprintf('Model loaded successfully.\n');

%% ============================================================
% 3. SELECT IMAGE
% =============================================================

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
% 4. READ IMAGE
% =============================================================

originalImage = imread(imageFile);

if size(originalImage,3) == 1

    originalImage = cat(3, ...
        originalImage, ...
        originalImage, ...
        originalImage);

end

%% ============================================================
% 5. PREPROCESS
% =============================================================

inputImage = imresize(originalImage,[224 224]);

inputImage = im2single(inputImage);

%% ============================================================
% 6. PREDICTION
% =============================================================

fprintf('\nRunning AI screening...\n');

scores = predict( ...
    trainedNet, ...
    inputImage);

if isa(scores,'dlarray')
    scores = extractdata(scores);
end

scores = gather(scores);

scores = squeeze(scores);

scores = double(scores(:));

%% ============================================================
% 7. CONVERT TO PROBABILITIES
% =============================================================

if all(scores >= 0) && ...
        abs(sum(scores)-1) < 1e-3

    probabilities = scores;

else

    probabilities = ...
        exp(scores - max(scores));

    probabilities = ...
        probabilities ./ sum(probabilities);

end

%% ============================================================
% 8. PREDICTED CLASS
% =============================================================

[confidence,classIndex] = ...
    max(probabilities);

predictedClass = ...
    string(classNames(classIndex));

%% ============================================================
% 9. DR GRADE
% =============================================================

switch predictedClass

    case "No DR"
        drGrade = 0;

    case "Mild DR"
        drGrade = 1;

    case "Moderate DR"
        drGrade = 2;

    case "Severe DR"
        drGrade = 3;

    case "Proliferative DR"
        drGrade = 4;

    otherwise
        drGrade = -1;

end

%% ============================================================
% 10. REFERABLE DECISION
% =============================================================

if drGrade >= 2

    referableStatus = "REFERABLE DR";

    recommendation = ...
        "Ophthalmology referral recommended.";

else

    referableStatus = "NON-REFERABLE DR";

    recommendation = ...
        "Routine diabetic eye screening follow-up recommended.";

end

%% ============================================================
% 11. SEVERITY DESCRIPTION
% =============================================================

switch drGrade

    case 0

        severityDescription = ...
            "No diabetic retinopathy detected.";

    case 1

        severityDescription = ...
            "Mild diabetic retinopathy.";

    case 2

        severityDescription = ...
            "Moderate diabetic retinopathy.";

    case 3

        severityDescription = ...
            "Severe diabetic retinopathy.";

    case 4

        severityDescription = ...
            "Proliferative diabetic retinopathy.";

    otherwise

        severityDescription = ...
            "DR severity could not be determined.";

end

%% ============================================================
% 12. DISPLAY COMMAND WINDOW RESULT
% =============================================================

fprintf('\n========================================\n');
fprintf('SCREENING RESULT\n');
fprintf('========================================\n');

fprintf('Predicted Class    : %s\n',predictedClass);
fprintf('DR Grade           : %d\n',drGrade);
fprintf('Confidence         : %.2f%%\n',confidence*100);
fprintf('Screening Category : %s\n',referableStatus);

fprintf('\n%s\n',severityDescription);

fprintf('\nRecommendation:\n%s\n',recommendation);

%% ============================================================
% 13. CREATE PROFESSIONAL REPORT
% =============================================================

fig = figure( ...
    'Name','AI-Assisted DR Screening Report', ...
    'NumberTitle','off', ...
    'Color',[0.96 0.97 0.98], ...
    'Toolbar','none', ...
    'MenuBar','none');

set(fig, ...
    'Position',[80 50 1450 900]);

%% ============================================================
% 14. HEADER
% =============================================================

header = axes( ...
    'Parent',fig, ...
    'Position',[0 0.90 1 0.10], ...
    'Color',[0.035 0.12 0.22], ...
    'XLim',[0 1], ...
    'YLim',[0 1], ...
    'XColor','none', ...
    'YColor','none');

hold(header,'on');

text(header, ...
    0.035,0.62, ...
    'DIABETIC RETINOPATHY', ...
    'Color','white', ...
    'FontSize',23, ...
    'FontWeight','bold', ...
    'VerticalAlignment','middle');

text(header, ...
    0.035,0.23, ...
    'AI-ASSISTED FUNDUS SCREENING REPORT', ...
    'Color',[0.75 0.82 0.90], ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'VerticalAlignment','middle');

text(header, ...
    0.965,0.50, ...
    datestr(now,'dd mmm yyyy  |  HH:MM'), ...
    'Color',[0.85 0.89 0.94], ...
    'FontSize',10, ...
    'HorizontalAlignment','right', ...
    'VerticalAlignment','middle');

hold(header,'off');

%% ============================================================
% 15. FUNDUS IMAGE PANEL
% =============================================================

imageAxes = axes( ...
    'Parent',fig, ...
    'Position',[0.045 0.49 0.40 0.36], ...
    'Color','white');

imshow(originalImage,'Parent',imageAxes);

title(imageAxes, ...
    'FUNDUS IMAGE', ...
    'FontSize',15, ...
    'FontWeight','bold');

%% ============================================================
% 16. SCREENING RESULT PANEL
% =============================================================

resultAxes = axes( ...
    'Parent',fig, ...
    'Position',[0.49 0.49 0.46 0.36], ...
    'Color','white', ...
    'XLim',[0 1], ...
    'YLim',[0 1], ...
    'XColor','none', ...
    'YColor','none');

hold(resultAxes,'on');

%% Result heading

text(resultAxes, ...
    0.06,0.88, ...
    'SCREENING RESULT', ...
    'FontSize',13, ...
    'FontWeight','bold');

%% Grade

text(resultAxes, ...
    0.06,0.68, ...
    sprintf('GRADE %d',drGrade), ...
    'FontSize',31, ...
    'FontWeight','bold');

%% Class

text(resultAxes, ...
    0.06,0.53, ...
    upper(char(predictedClass)), ...
    'FontSize',18, ...
    'FontWeight','bold');

%% Confidence

text(resultAxes, ...
    0.06,0.37, ...
    sprintf('Model confidence   %.1f%%', ...
    confidence*100), ...
    'FontSize',13);

%% Confidence bar background

rectangle(resultAxes, ...
    'Position',[0.06 0.27 0.82 0.035], ...
    'FaceColor',[0.88 0.90 0.93], ...
    'EdgeColor','none');

rectangle(resultAxes, ...
    'Position',[0.06 0.27 0.82*confidence 0.035], ...
    'FaceColor',[0.10 0.45 0.70], ...
    'EdgeColor','none');

%% Referable status

if drGrade >= 2

    statusTextColor = [0.70 0.12 0.10];

else

    statusTextColor = [0.10 0.48 0.28];

end

text(resultAxes, ...
    0.06,0.12, ...
    char(referableStatus), ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'Color',statusTextColor);

hold(resultAxes,'off');

%% ============================================================
% 17. PROBABILITY PANEL
% =============================================================

probAxes = axes( ...
    'Parent',fig, ...
    'Position',[0.045 0.10 0.40 0.30], ...
    'Color','white', ...
    'XColor','none');

hold(probAxes,'on');

%% Horizontal probability bars

yPositions = 5:-1:1;

barHeight = 0.52;

for i = 1:numel(classNames)

    probability = probabilities(i)*100;

    % Background

    rectangle(probAxes, ...
        'Position',[0 yPositions(i)-barHeight/2 ...
        100 barHeight], ...
        'FaceColor',[0.91 0.92 0.94], ...
        'EdgeColor','none');

    % Probability

    rectangle(probAxes, ...
        'Position',[0 yPositions(i)-barHeight/2 ...
        probability barHeight], ...
        'FaceColor',[0.15 0.48 0.72], ...
        'EdgeColor','none');

    % Class name

    text(probAxes, ...
        -3,yPositions(i), ...
        char(classNames(i)), ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'FontSize',11, ...
        'FontWeight','bold');

    % Percentage

    text(probAxes, ...
        min(probability+2,96), ...
        yPositions(i), ...
        sprintf('%.1f%%',probability), ...
        'VerticalAlignment','middle', ...
        'FontSize',10, ...
        'FontWeight','bold');

end

xlim(probAxes,[-45 110]);

ylim(probAxes,[0.3 5.7]);

title(probAxes, ...
    'MODEL PROBABILITY DISTRIBUTION', ...
    'FontSize',13, ...
    'FontWeight','bold');

hold(probAxes,'off');

%% ============================================================
% 18. RECOMMENDATION PANEL
% =============================================================

recommendAxes = axes( ...
    'Parent',fig, ...
    'Position',[0.49 0.10 0.46 0.30], ...
    'Color','white', ...
    'XLim',[0 1], ...
    'YLim',[0 1], ...
    'XColor','none', ...
    'YColor','none');

hold(recommendAxes,'on');

%% Heading

text(recommendAxes, ...
    0.06,0.87, ...
    'CLINICAL SCREENING RECOMMENDATION', ...
    'FontSize',13, ...
    'FontWeight','bold');

%% Severity

text(recommendAxes, ...
    0.06,0.66, ...
    char(severityDescription), ...
    'FontSize',15, ...
    'FontWeight','bold');

%% Recommendation

text(recommendAxes, ...
    0.06,0.48, ...
    char(recommendation), ...
    'FontSize',13);

%% Clinical note

text(recommendAxes, ...
    0.06,0.25, ...
    'AI-assisted screening result.', ...
    'FontSize',11, ...
    'FontWeight','bold');

text(recommendAxes, ...
    0.06,0.14, ...
    'Final clinical assessment should be performed by a qualified eye-care professional.', ...
    'FontSize',10);

hold(recommendAxes,'off');

%% ============================================================
% 19. FOOTER
% =============================================================

footer = axes( ...
    'Parent',fig, ...
    'Position',[0.03 0.015 0.94 0.045], ...
    'Color',[0.90 0.92 0.95], ...
    'XLim',[0 1], ...
    'YLim',[0 1], ...
    'XColor','none', ...
    'YColor','none');

text(footer, ...
    0.01,0.50, ...
    'Explainable AI for Diabetic Retinopathy Screening', ...
    'FontSize',9, ...
    'Color',[0.25 0.30 0.35], ...
    'VerticalAlignment','middle');

text(footer, ...
    0.99,0.50, ...
    'Prototype | Clinical validation required', ...
    'FontSize',9, ...
    'Color',[0.25 0.30 0.35], ...
    'HorizontalAlignment','right', ...
    'VerticalAlignment','middle');

%% ============================================================
% 20. SAVE PROFESSIONAL REPORT
% =============================================================

[~,baseName,~] = ...
    fileparts(fileName);

reportImage = fullfile( ...
    resultsFolder, ...
    [baseName '_Professional_DR_Report.png']);

exportgraphics( ...
    fig, ...
    reportImage, ...
    'Resolution',200);

%% ============================================================
% 21. SAVE REPORT DATA
% =============================================================

reportData.imageName = fileName;
reportData.predictedClass = predictedClass;
reportData.drGrade = drGrade;
reportData.confidence = confidence;
reportData.probabilities = probabilities;
reportData.classNames = classNames;
reportData.referableStatus = referableStatus;
reportData.recommendation = recommendation;
reportData.severityDescription = severityDescription;
reportData.generatedAt = datetime('now');

reportFile = fullfile( ...
    resultsFolder, ...
    [baseName '_Professional_DR_Report.mat']);

save(reportFile,'reportData');

%% ============================================================
% 22. COMMAND WINDOW SUMMARY
% =============================================================

fprintf('\n========================================\n');
fprintf('PROFESSIONAL REPORT GENERATED\n');
fprintf('========================================\n');

fprintf('\nDR Grade       : %d\n',drGrade);
fprintf('Classification : %s\n',predictedClass);
fprintf('Confidence     : %.2f%%\n',confidence*100);
fprintf('Status         : %s\n',referableStatus);

fprintf('\nRecommendation:\n');
fprintf('%s\n',recommendation);

fprintf('\nReport image:\n');
fprintf('%s\n',reportImage);

fprintf('\nReport data:\n');
fprintf('%s\n',reportFile);

fprintf('\n========================================\n');
fprintf('SCREENING REPORT COMPLETE\n');
fprintf('========================================\n');