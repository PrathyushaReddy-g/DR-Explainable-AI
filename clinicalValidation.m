function clinicalValidation
% =========================================================================
% CLINICAL VALIDATION DASHBOARD
% Explainable AI for Diabetic Retinopathy Screening
%
% MATLAB R2026a
% Human-in-the-Loop Clinical Validation Prototype
% =========================================================================

clc;

%% ========================================================================
% PROJECT PATHS
% =========================================================================

projectRoot = fileparts(mfilename('fullpath'));

modelFile  = fullfile(projectRoot,'models','DR_Model.mat');
resultsDir = fullfile(projectRoot,'results');

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end

%% ========================================================================
% COLORS
% =========================================================================

NAVY      = [0.025 0.110 0.190];
NAVY2     = [0.045 0.170 0.280];

BLUE      = [0.035 0.420 0.760];
BLUE2     = [0.055 0.500 0.800];

WHITE     = [1.000 1.000 1.000];
BG        = [0.945 0.960 0.975];

TEXT      = [0.055 0.120 0.190];
MUTED     = [0.400 0.460 0.530];

GREEN     = [0.020 0.500 0.250];
ORANGE    = [0.900 0.500 0.050];
RED       = [0.800 0.080 0.100];

LIGHTBLUE = [0.650 0.820 0.970];
LIGHTGRAY = [0.920 0.935 0.950];
BLUEBG    = [0.930 0.960 0.990];

%% ========================================================================
% STATE
% =========================================================================

state = struct();

state.image = [];
state.fileName = "";

state.aiGrade = "Not analyzed";
state.aiGradeIndex = -1;
state.aiConfidence = 0;
state.aiReferable = false;
state.aiProbabilities = zeros(5,1);

state.referenceGrade = "";
state.referenceGradeIndex = -1;
state.referenceReferable = false;

state.classNames = [ ...
    "No DR", ...
    "Mild DR", ...
    "Moderate DR", ...
    "Severe DR", ...
    "Proliferative DR"];

state.trainedNet = [];
state.modelLoaded = false;

%% ========================================================================
% LOAD RESNET-18 MODEL
% =========================================================================

if isfile(modelFile)

    try

        D = load(modelFile);

        if isfield(D,'trainedNet')
            state.trainedNet = D.trainedNet;
            state.modelLoaded = true;
        end

        if isfield(D,'classNames')

            names = string(D.classNames);

            if numel(names) >= 5
                state.classNames = names(1:5);
            end

        end

    catch ME

        warning("Model loading failed: %s",ME.message);

    end

end

%% ========================================================================
% FIGURE SIZE
% =========================================================================

screenSize = get(groot,'ScreenSize');

screenW = screenSize(3);
screenH = screenSize(4);

figW = min(1450,screenW - 80);
figH = min(850,screenH - 100);

figW = max(figW,1100);
figH = max(figH,680);

figX = max(20,(screenW-figW)/2);
figY = max(30,(screenH-figH)/2);

%% ========================================================================
% MAIN WINDOW
% =========================================================================

fig = uifigure( ...
    'Name','Clinical Validation | Diabetic Retinopathy', ...
    'Color',BG, ...
    'Position',[figX figY figW figH]);

fig.AutoResizeChildren = 'off';

%% ========================================================================
% HEADER
% =========================================================================

headerH = 82;

header = uipanel(fig);

header.Position = ...
    [0 figH-headerH figW headerH];

header.BackgroundColor = NAVY;
header.BorderType = 'none';

%% DR LOGO

logo = uipanel(header);

logo.Position = [24 15 64 52];

logo.BackgroundColor = BLUE;
logo.BorderType = 'none';

logoText = uilabel(logo);

logoText.Text = "DR";

logoText.FontSize = 21;
logoText.FontWeight = 'bold';
logoText.FontColor = WHITE;

logoText.HorizontalAlignment = 'center';
logoText.VerticalAlignment = 'center';

logoText.Position = [0 0 64 52];

%% TITLE

titleText = uilabel(header);

titleText.Text = ...
    "DIABETIC RETINOPATHY SCREENING";

titleText.FontSize = 19;
titleText.FontWeight = 'bold';
titleText.FontColor = WHITE;

titleText.Position = ...
    [108 43 600 30];

%% SUBTITLE

subtitleText = uilabel(header);

subtitleText.Text = ...
    "Clinical Validation  |  Explainable AI";

subtitleText.FontSize = 11;
subtitleText.FontWeight = 'bold';
subtitleText.FontColor = LIGHTBLUE;

subtitleText.Position = ...
    [108 17 500 24];

%% MODEL STATUS

modelStatus = uilabel(header);

if state.modelLoaded

    modelStatus.Text = ...
        "●  RESNET-18 MODEL READY";

    modelStatus.FontColor = ...
        [0.300 0.900 0.500];

else

    modelStatus.Text = ...
        "●  MODEL NOT FOUND";

    modelStatus.FontColor = ...
        [1.000 0.350 0.350];

end

modelStatus.FontSize = 11;
modelStatus.FontWeight = 'bold';

modelStatus.HorizontalAlignment = 'right';

modelStatus.Position = ...
    [figW-410 43 380 24];

%% PROTOTYPE TEXT

prototypeText = uilabel(header);

prototypeText.Text = ...
    "Research Prototype  •  Human-in-the-Loop Review";

prototypeText.FontSize = 9;
prototypeText.FontColor = LIGHTBLUE;

prototypeText.HorizontalAlignment = 'right';

prototypeText.Position = ...
    [figW-500 17 470 20];

%% ========================================================================
% WORKFLOW
% =========================================================================

workflowH = 55;

workflow = uipanel(fig);

workflow.Position = ...
    [0 figH-headerH-workflowH figW workflowH];

workflow.BackgroundColor = WHITE;
workflow.BorderType = 'none';

%% STEP 1

makeStep( ...
    workflow, ...
    [35 10 34 34], ...
    "1", ...
    BLUE);

makeText( ...
    workflow, ...
    [78 12 130 30], ...
    "FUNDUS IMAGE", ...
    TEXT);

makeArrow( ...
    workflow, ...
    [208 8 40 35]);

%% STEP 2

makeStep( ...
    workflow, ...
    [260 10 34 34], ...
    "2", ...
    BLUE);

makeText( ...
    workflow, ...
    [303 12 130 30], ...
    "AI SCREENING", ...
    TEXT);

makeArrow( ...
    workflow, ...
    [433 8 40 35]);

%% STEP 3

makeStep( ...
    workflow, ...
    [485 10 34 34], ...
    "3", ...
    BLUE);

makeText( ...
    workflow, ...
    [528 12 165 30], ...
    "CLINICAL REFERENCE", ...
    TEXT);

makeArrow( ...
    workflow, ...
    [695 8 40 35]);

%% STEP 4

makeStep( ...
    workflow, ...
    [748 10 34 34], ...
    "4", ...
    GREEN);

makeText( ...
    workflow, ...
    [791 12 175 30], ...
    "AGREEMENT ANALYSIS", ...
    TEXT);

%% WORKFLOW NOTE

workflowNote = uilabel(workflow);

workflowNote.Text = ...
    "Human reference required for validation";

workflowNote.FontSize = 9;
workflowNote.FontColor = MUTED;

workflowNote.HorizontalAlignment = 'right';

workflowNote.Position = ...
    [figW-350 12 320 30];

%% ========================================================================
% MAIN LAYOUT
% =========================================================================

bottomH = 42;
resultH = 175;

contentTop = ...
    figH-headerH-workflowH;

resultY = bottomH + 12;

contentY = ...
    resultY + resultH + 12;

contentH = ...
    contentTop-contentY-12;

leftMargin = 25;
rightMargin = 25;

gap = 16;

availableW = ...
    figW-leftMargin-rightMargin-2*gap;

imageW = round(availableW*0.405);

aiW = round(availableW*0.285);

refW = ...
    availableW-imageW-aiW;

imageX = leftMargin;

aiX = ...
    imageX+imageW+gap;

refX = ...
    aiX+aiW+gap;

%% ========================================================================
% 01 FUNDUS IMAGE
% =========================================================================

imageCard = uipanel(fig);

imageCard.Position = ...
    [imageX contentY imageW contentH];

imageCard.BackgroundColor = WHITE;
imageCard.BorderType = 'line';

%% TITLE

imageTitle = uilabel(imageCard);

imageTitle.Text = ...
    "01   FUNDUS IMAGE";

imageTitle.FontSize = 15;
imageTitle.FontWeight = 'bold';
imageTitle.FontColor = TEXT;

imageTitle.Position = ...
    [20 contentH-45 imageW-40 28];

%% SUBTITLE

imageSubtitle = uilabel(imageCard);

imageSubtitle.Text = ...
    "Retinal image selected for AI-assisted screening";

imageSubtitle.FontSize = 9;
imageSubtitle.FontColor = MUTED;

imageSubtitle.Position = ...
    [20 contentH-69 imageW-40 20];

%% IMAGE AXES

imageAxes = uiaxes(imageCard);

axesW = imageW-44;
axesH = contentH-145;

imageAxes.Position = ...
    [22 65 axesW axesH];

imageAxes.BackgroundColor = ...
    [0.965 0.970 0.975];

imageAxes.XTick = [];
imageAxes.YTick = [];

imageAxes.Box = 'on';

title(imageAxes,"Upload a fundus image");

try
    imageAxes.Toolbar.Visible = 'off';
catch
end

%% FILE NAME

fileLabel = uilabel(imageCard);

fileLabel.Text = ...
    "No image selected";

fileLabel.FontSize = 9;
fileLabel.FontColor = MUTED;

fileLabel.Position = ...
    [22 30 250 22];

%% UPLOAD BUTTON

uploadButton = uibutton(imageCard);

uploadButton.Text = ...
    "＋  Upload Fundus Image";

uploadButton.FontSize = 10;
uploadButton.FontWeight = 'bold';

uploadButton.FontColor = WHITE;
uploadButton.BackgroundColor = BLUE2;

uploadButton.Position = ...
    [imageW-205 22 180 38];

uploadButton.ButtonPushedFcn = @uploadImage;

%% ========================================================================
% 02 AI SCREENING OUTPUT
% =========================================================================

aiCard = uipanel(fig);

aiCard.Position = ...
    [aiX contentY aiW contentH];

aiCard.BackgroundColor = WHITE;
aiCard.BorderType = 'line';

%% TITLE

aiTitle = uilabel(aiCard);

aiTitle.Text = ...
    "02   AI SCREENING OUTPUT";

aiTitle.FontSize = 15;
aiTitle.FontWeight = 'bold';
aiTitle.FontColor = TEXT;

aiTitle.Position = ...
    [20 contentH-45 aiW-40 28];

%% SUBTITLE

aiSubtitle = uilabel(aiCard);

aiSubtitle.Text = ...
    "ResNet-18 classification result";

aiSubtitle.FontSize = 9;
aiSubtitle.FontColor = MUTED;

aiSubtitle.Position = ...
    [20 contentH-69 aiW-40 20];

%% GRADE PANEL

gradePanelH = 92;

gradePanel = uipanel(aiCard);

gradePanel.Position = ...
    [20 contentH-175 aiW-40 gradePanelH];

gradePanel.BackgroundColor = BLUEBG;
gradePanel.BorderType = 'none';

%% GRADE TITLE

gradeTitle = uilabel(gradePanel);

gradeTitle.Text = ...
    "AI PREDICTED GRADE";

gradeTitle.FontSize = 8;
gradeTitle.FontWeight = 'bold';
gradeTitle.FontColor = MUTED;

gradeTitle.Position = ...
    [14 53 aiW-68 20];

%% AI GRADE

aiGradeLabel = uilabel(gradePanel);

aiGradeLabel.Text = "—";

aiGradeLabel.FontSize = 19;
aiGradeLabel.FontWeight = 'bold';
aiGradeLabel.FontColor = BLUE;

aiGradeLabel.Position = ...
    [14 13 aiW-68 38];

%% CONFIDENCE

confidenceTitle = uilabel(aiCard);

confidenceTitle.Text = ...
    "CONFIDENCE";

confidenceTitle.FontSize = 8;
confidenceTitle.FontWeight = 'bold';
confidenceTitle.FontColor = MUTED;

confidenceTitle.Position = ...
    [20 contentH-205 130 20];

aiConfidenceLabel = uilabel(aiCard);

aiConfidenceLabel.Text = "—";

aiConfidenceLabel.FontSize = 12;
aiConfidenceLabel.FontWeight = 'bold';
aiConfidenceLabel.FontColor = TEXT;

aiConfidenceLabel.HorizontalAlignment = 'right';

aiConfidenceLabel.Position = ...
    [aiW-130 contentH-208 110 25];

%% REFERABLE DR

referableTitle = uilabel(aiCard);

referableTitle.Text = ...
    "REFERABLE DR";

referableTitle.FontSize = 8;
referableTitle.FontWeight = 'bold';
referableTitle.FontColor = MUTED;

referableTitle.Position = ...
    [20 contentH-238 130 20];

aiReferableLabel = uilabel(aiCard);

aiReferableLabel.Text = "—";

aiReferableLabel.FontSize = 12;
aiReferableLabel.FontWeight = 'bold';
aiReferableLabel.FontColor = TEXT;

aiReferableLabel.HorizontalAlignment = 'right';

aiReferableLabel.Position = ...
    [aiW-180 contentH-241 160 25];

%% AI STATUS

aiStatusLabel = uilabel(aiCard);

aiStatusLabel.Text = ...
    "READY FOR ANALYSIS";

aiStatusLabel.FontSize = 8;
aiStatusLabel.FontWeight = 'bold';
aiStatusLabel.FontColor = MUTED;

aiStatusLabel.HorizontalAlignment = 'center';

aiStatusLabel.Position = ...
    [20 66 aiW-40 20];

%% ========================================================================
% RUN AI SCREENING BUTTON
%
% IMPORTANT CHANGE:
% Button moved lower to Y = 8.
% This keeps it separated from the REFERABLE DR information.
% =========================================================================

analyzeButton = uibutton(aiCard);

analyzeButton.Text = ...
    "▶  RUN AI SCREENING";

analyzeButton.FontSize = 10;
analyzeButton.FontWeight = 'bold';

analyzeButton.FontColor = WHITE;
analyzeButton.BackgroundColor = BLUE2;

% LOWER POSITION
analyzeButton.Position = ...
    [35 8 aiW-70 40];

analyzeButton.ButtonPushedFcn = @runAI;

%% ========================================================================
% 03 CLINICAL REFERENCE
% =========================================================================

referenceCard = uipanel(fig);

referenceCard.Position = ...
    [refX contentY refW contentH];

referenceCard.BackgroundColor = NAVY;
referenceCard.BorderType = 'none';

%% TITLE

referenceTitle = uilabel(referenceCard);

referenceTitle.Text = ...
    "03   CLINICAL REFERENCE";

referenceTitle.FontSize = 15;
referenceTitle.FontWeight = 'bold';
referenceTitle.FontColor = WHITE;

referenceTitle.Position = ...
    [20 contentH-45 refW-40 28];

%% SUBTITLE

referenceSubtitle = uilabel(referenceCard);

referenceSubtitle.Text = ...
    "Human / clinical reviewer reference standard";

referenceSubtitle.FontSize = 9;
referenceSubtitle.FontColor = LIGHTBLUE;

referenceSubtitle.Position = ...
    [20 contentH-69 refW-40 20];

%% REFERENCE TITLE

referenceGradeTitle = uilabel(referenceCard);

referenceGradeTitle.Text = ...
    "REFERENCE DR GRADE";

referenceGradeTitle.FontSize = 8;
referenceGradeTitle.FontWeight = 'bold';
referenceGradeTitle.FontColor = LIGHTBLUE;

referenceGradeTitle.Position = ...
    [24 contentH-130 refW-48 20];

%% DROPDOWN

referenceDropDown = uidropdown(referenceCard);

referenceDropDown.Items = ...
    cellstr(state.classNames);

referenceDropDown.ItemsData = ...
    0:4;

referenceDropDown.Value = 0;

referenceDropDown.FontSize = 11;

referenceDropDown.Position = ...
    [24 contentH-185 refW-48 38];

%% REFERRAL THRESHOLD

thresholdLabel = uilabel(referenceCard);

thresholdLabel.Text = ...
    "Referral threshold: Grade ≥ 2";

thresholdLabel.FontSize = 8;
thresholdLabel.FontColor = LIGHTBLUE;

thresholdLabel.HorizontalAlignment = 'center';

thresholdLabel.Position = ...
    [24 contentH-220 refW-48 20];

%% COMPARE BUTTON

validateButton = uibutton(referenceCard);

validateButton.Text = ...
    "✓  COMPARE AI VS REFERENCE";

validateButton.FontSize = 10;
validateButton.FontWeight = 'bold';

validateButton.FontColor = WHITE;
validateButton.BackgroundColor = BLUE2;

validateButton.Position = ...
    [30 62 refW-60 42];

validateButton.ButtonPushedFcn = @validateResult;

%% REFERENCE STATUS

referenceStatus = uilabel(referenceCard);

referenceStatus.Text = ...
    "REFERENCE NOT ENTERED";

referenceStatus.FontSize = 9;
referenceStatus.FontWeight = 'bold';
referenceStatus.FontColor = LIGHTBLUE;

referenceStatus.HorizontalAlignment = 'center';

referenceStatus.Position = ...
    [20 25 refW-40 22];

%% ========================================================================
% 04 VALIDATION RESULT
% =========================================================================

resultCard = uipanel(fig);

resultW = ...
    figW-leftMargin-rightMargin;

resultCard.Position = ...
    [leftMargin resultY resultW resultH];

resultCard.BackgroundColor = WHITE;
resultCard.BorderType = 'line';

%% TITLE

resultTitle = uilabel(resultCard);

resultTitle.Text = ...
    "04   VALIDATION RESULT";

resultTitle.FontSize = 14;
resultTitle.FontWeight = 'bold';
resultTitle.FontColor = TEXT;

resultTitle.Position = ...
    [20 resultH-43 resultW-40 25];

%% SUBTITLE

resultSubtitle = uilabel(resultCard);

resultSubtitle.Text = ...
    "Comparison between AI screening and human / clinical reference";

resultSubtitle.FontSize = 8;
resultSubtitle.FontColor = MUTED;

resultSubtitle.Position = ...
    [20 resultH-67 resultW-40 18];

%% RESULT BOX SIZES

boxGap = 14;

boxW = floor( ...
    (resultW-40-3*boxGap)/4);

boxY = 38;
boxH = 68;

box1X = 20;

box2X = ...
    box1X+boxW+boxGap;

box3X = ...
    box2X+boxW+boxGap;

box4X = ...
    box3X+boxW+boxGap;

%% BOX 1

createResultBox( ...
    resultCard, ...
    [box1X boxY boxW boxH], ...
    "AI RESULT");

%% BOX 2

createResultBox( ...
    resultCard, ...
    [box2X boxY boxW boxH], ...
    "CLINICAL REFERENCE");

%% BOX 3

createResultBox( ...
    resultCard, ...
    [box3X boxY boxW boxH], ...
    "GRADE AGREEMENT");

%% BOX 4

createResultBox( ...
    resultCard, ...
    [box4X boxY boxW boxH], ...
    "REFERRAL AGREEMENT");

%% AI DECISION

aiDecisionLabel = uilabel(resultCard);

aiDecisionLabel.Text = "—";

aiDecisionLabel.FontSize = 11;
aiDecisionLabel.FontWeight = 'bold';
aiDecisionLabel.FontColor = TEXT;

aiDecisionLabel.Position = ...
    [box1X+12 boxY+8 boxW-24 28];

%% REFERENCE DECISION

referenceDecisionLabel = uilabel(resultCard);

referenceDecisionLabel.Text = "—";

referenceDecisionLabel.FontSize = 11;
referenceDecisionLabel.FontWeight = 'bold';
referenceDecisionLabel.FontColor = TEXT;

referenceDecisionLabel.Position = ...
    [box2X+12 boxY+8 boxW-24 28];

%% GRADE AGREEMENT

agreementLabel = uilabel(resultCard);

agreementLabel.Text = "—";

agreementLabel.FontSize = 11;
agreementLabel.FontWeight = 'bold';
agreementLabel.FontColor = MUTED;

agreementLabel.HorizontalAlignment = 'center';

agreementLabel.Position = ...
    [box3X+5 boxY+8 boxW-10 28];

%% REFERRAL AGREEMENT

referralAgreementLabel = uilabel(resultCard);

referralAgreementLabel.Text = "—";

referralAgreementLabel.FontSize = 11;
referralAgreementLabel.FontWeight = 'bold';
referralAgreementLabel.FontColor = MUTED;

referralAgreementLabel.HorizontalAlignment = 'center';

referralAgreementLabel.Position = ...
    [box4X+5 boxY+8 boxW-10 28];

%% DISCLAIMER

disclaimer = uilabel(resultCard);

disclaimer.Text = ...
    "Research prototype  •  Reference grading should be performed by an appropriately trained clinical reviewer  •  AI output does not replace ophthalmologist examination.";

disclaimer.FontSize = 7;
disclaimer.FontColor = MUTED;

disclaimer.HorizontalAlignment = 'center';

disclaimer.Position = ...
    [20 10 resultW-40 18];

%% ========================================================================
% FOOTER
% =========================================================================

footer = uipanel(fig);

footer.Position = ...
    [0 0 figW bottomH];

footer.BackgroundColor = NAVY2;
footer.BorderType = 'none';

footerLabel = uilabel(footer);

footerLabel.Text = ...
    "AI-assisted screening  |  ResNet-18  |  5-class DR grading  |  Referable threshold: Moderate DR or higher  |  Human review required";

footerLabel.FontSize = 8;
footerLabel.FontWeight = 'bold';
footerLabel.FontColor = LIGHTBLUE;

footerLabel.HorizontalAlignment = 'center';
footerLabel.VerticalAlignment = 'center';

footerLabel.Position = ...
    [20 5 figW-40 bottomH-10];

%% ========================================================================
% UPLOAD IMAGE
% =========================================================================

    function uploadImage(~,~)

        [file,path] = uigetfile( ...
            {'*.png;*.jpg;*.jpeg;*.tif;*.tiff','Fundus Images'; ...
             '*.*','All Files'}, ...
            'Select Retinal Fundus Image');

        if isequal(file,0)
            return;
        end

        try

            img = imread(fullfile(path,file));

            %% Convert grayscale to RGB

            if ndims(img) == 2
                img = repmat(img,[1 1 3]);
            end

            %% Remove alpha channel

            if size(img,3) > 3
                img = img(:,:,1:3);
            end

            %% Convert to uint8

            if ~isa(img,'uint8')
                img = im2uint8(img);
            end

            state.image = img;

            state.fileName = string(file);

            %% Display

            imshow( ...
                state.image, ...
                'Parent',imageAxes);

            axis(imageAxes,'image');

            imageAxes.XTick = [];
            imageAxes.YTick = [];

            title( ...
                imageAxes, ...
                "Selected Fundus Image");

            fileLabel.Text = ...
                char(state.fileName);

            resetValidation();

            aiStatusLabel.Text = ...
                "IMAGE READY FOR ANALYSIS";

            aiStatusLabel.FontColor = BLUE;

        catch ME

            uialert( ...
                fig, ...
                ['Unable to load image.' newline newline ...
                ME.message], ...
                'Image Error');

        end

    end

%% ========================================================================
% RUN AI
% =========================================================================

    function runAI(~,~)

        if isempty(state.image)

            uialert( ...
                fig, ...
                'Please upload a fundus image first.', ...
                'No Image');

            return;

        end

        if ~state.modelLoaded

            uialert( ...
                fig, ...
                ['The trained ResNet-18 model was not found.' ...
                newline newline ...
                'Expected location:' newline ...
                modelFile], ...
                'Model Error');

            return;

        end

        try

            %% Status

            aiStatusLabel.Text = ...
                "●  RUNNING RESNET-18...";

            aiStatusLabel.FontColor = ORANGE;

            analyzeButton.Enable = 'off';

            drawnow;

            %% Classification

            [grade, ...
             gradeIndex, ...
             confidence, ...
             probs] = ...
                classifyForValidation( ...
                state.image, ...
                state.trainedNet, ...
                state.classNames);

            %% Store

            state.aiGrade = grade;

            state.aiGradeIndex = gradeIndex;

            state.aiConfidence = confidence;

            state.aiProbabilities = probs;

            state.aiReferable = ...
                gradeIndex >= 2;

            %% Grade

            aiGradeLabel.Text = ...
                char(state.aiGrade);

            aiGradeLabel.FontColor = ...
                gradeColor(state.aiGradeIndex);

            %% Confidence

            aiConfidenceLabel.Text = ...
                sprintf("%.1f%%", ...
                100*state.aiConfidence);

            %% Referable

            if state.aiReferable

                aiReferableLabel.Text = ...
                    "YES  •  REVIEW";

                aiReferableLabel.FontColor = RED;

            else

                aiReferableLabel.Text = ...
                    "NO";

                aiReferableLabel.FontColor = GREEN;

            end

            %% Status

            aiStatusLabel.Text = ...
                "●  AI SCREENING COMPLETED";

            aiStatusLabel.FontColor = GREEN;

            %% Reset old validation result

            aiDecisionLabel.Text = "—";
            aiDecisionLabel.FontColor = TEXT;

            referenceDecisionLabel.Text = "—";
            referenceDecisionLabel.FontColor = TEXT;

            agreementLabel.Text = "—";
            agreementLabel.FontColor = MUTED;

            referralAgreementLabel.Text = "—";
            referralAgreementLabel.FontColor = MUTED;

            referenceStatus.Text = ...
                "REFERENCE READY FOR REVIEW";

            analyzeButton.Enable = 'on';

            drawnow;

        catch ME

            analyzeButton.Enable = 'on';

            aiStatusLabel.Text = ...
                "●  AI ANALYSIS FAILED";

            aiStatusLabel.FontColor = RED;

            uialert( ...
                fig, ...
                ['AI analysis failed.' newline newline ...
                ME.message], ...
                'AI Error');

        end

    end

%% ========================================================================
% COMPARE AI VS CLINICAL REFERENCE
% =========================================================================

    function validateResult(~,~)

        if state.aiGradeIndex < 0

            uialert( ...
                fig, ...
                'Run AI screening before comparing with the clinical reference.', ...
                'Validation');

            return;

        end

        %% Reference grade

        referenceIndex = ...
            referenceDropDown.Value;

        state.referenceGradeIndex = ...
            referenceIndex;

        state.referenceGrade = ...
            state.classNames(referenceIndex+1);

        state.referenceReferable = ...
            referenceIndex >= 2;

        %% AI result

        aiDecisionLabel.Text = ...
            char(state.aiGrade);

        aiDecisionLabel.FontColor = ...
            gradeColor(state.aiGradeIndex);

        %% Clinical reference

        referenceDecisionLabel.Text = ...
            char(state.referenceGrade);

        referenceDecisionLabel.FontColor = ...
            gradeColor(state.referenceGradeIndex);

        %% Grade agreement

        gradeAgreement = ...
            state.aiGradeIndex == ...
            state.referenceGradeIndex;

        if gradeAgreement

            agreementLabel.Text = ...
                "✓  MATCH";

            agreementLabel.FontColor = GREEN;

        else

            agreementLabel.Text = ...
                "✕  MISMATCH";

            agreementLabel.FontColor = RED;

        end

        %% Referral agreement

        referralAgreement = ...
            state.aiReferable == ...
            state.referenceReferable;

        if referralAgreement

            referralAgreementLabel.Text = ...
                "✓  MATCH";

            referralAgreementLabel.FontColor = GREEN;

        else

            referralAgreementLabel.Text = ...
                "✕  MISMATCH";

            referralAgreementLabel.FontColor = RED;

        end

        %% Status

        referenceStatus.Text = ...
            ['REFERENCE: ' ...
            char(state.referenceGrade)];

        %% Save

        saveValidationRecord( ...
            gradeAgreement, ...
            referralAgreement);

    end

%% ========================================================================
% RESNET-18 CLASSIFICATION
% =========================================================================

    function [grade,gradeIndex,confidence,probs] = ...
            classifyForValidation(img,net,classNames)

        %% Default input size

        inputSize = [224 224 3];

        try

            if isa(net,'dlnetwork')

                inputSize = ...
                    net.Layers(1).InputSize;

            end

        catch

            inputSize = [224 224 3];

        end

        if numel(inputSize) < 3

            inputSize = ...
                [inputSize(1) inputSize(2) 3];

        end

        %% Resize

        x = imresize( ...
            img, ...
            inputSize(1:2));

        %% RGB

        if ndims(x) == 2

            x = repmat(x,[1 1 3]);

        end

        if size(x,3) > 3

            x = x(:,:,1:3);

        end

        %% Single

        x = im2single(x);

        %% Predict

        scores = predict(net,x);

        %% Extract data

        if isa(scores,'dlarray')

            scores = extractdata(scores);

        end

        scores = gather(scores);

        scores = double(scores);

        scores = squeeze(scores);

        scores = scores(:);

        %% Verify classes

        if numel(scores) < 5

            error( ...
                "The trained network returned fewer than five class scores.");

        end

        scores = scores(1:5);

        %% Convert to probabilities

        if all(isfinite(scores)) && ...
                all(scores >= 0) && ...
                abs(sum(scores)-1) < 0.05

            probs = ...
                scores ./ max(sum(scores),eps);

        else

            scores = ...
                scores-max(scores);

            expScores = ...
                exp(scores);

            probs = ...
                expScores ./ max(sum(expScores),eps);

        end

        probs = ...
            double(probs(:));

        %% Find predicted class

        [confidence,classIndex] = ...
            max(probs);

        classIndex = ...
            max(1,min(5,classIndex));

        gradeIndex = ...
            classIndex-1;

        grade = ...
            string(classNames(classIndex));

    end

%% ========================================================================
% SAVE VALIDATION RECORD
% =========================================================================

    function saveValidationRecord( ...
            gradeAgreement, ...
            referralAgreement)

        try

            timestamp = datetime( ...
                "now", ...
                "Format","yyyy-MM-dd HH:mm:ss");

            csvFile = ...
                fullfile( ...
                resultsDir, ...
                "clinical_validation_records.csv");

            record = table( ...
                string(timestamp), ...
                state.fileName, ...
                state.aiGrade, ...
                100*state.aiConfidence, ...
                string(yesNo(state.aiReferable)), ...
                state.referenceGrade, ...
                string(yesNo(state.referenceReferable)), ...
                string(yesNo(gradeAgreement)), ...
                string(yesNo(referralAgreement)), ...
                'VariableNames',{ ...
                'Timestamp', ...
                'Image', ...
                'AI_Grade', ...
                'AI_Confidence_Percent', ...
                'AI_Referable', ...
                'Clinical_Reference', ...
                'Reference_Referable', ...
                'Grade_Agreement', ...
                'Referral_Agreement'});

            if isfile(csvFile)

                oldData = ...
                    readtable(csvFile);

                writetable( ...
                    [oldData;record], ...
                    csvFile);

            else

                writetable( ...
                    record, ...
                    csvFile);

            end

        catch ME

            warning( ...
                "Validation record could not be saved: %s", ...
                ME.message);

        end

    end

%% ========================================================================
% RESET
% =========================================================================

    function resetValidation()

        state.aiGrade = ...
            "Not analyzed";

        state.aiGradeIndex = -1;

        state.aiConfidence = 0;

        state.aiReferable = false;

        state.referenceGrade = "";

        state.referenceGradeIndex = -1;

        state.referenceReferable = false;

        %% AI

        aiGradeLabel.Text = "—";

        aiGradeLabel.FontColor = BLUE;

        aiConfidenceLabel.Text = "—";

        aiReferableLabel.Text = "—";

        aiReferableLabel.FontColor = TEXT;

        aiStatusLabel.Text = ...
            "READY FOR ANALYSIS";

        aiStatusLabel.FontColor = MUTED;

        %% Validation

        aiDecisionLabel.Text = "—";
        aiDecisionLabel.FontColor = TEXT;

        referenceDecisionLabel.Text = "—";
        referenceDecisionLabel.FontColor = TEXT;

        agreementLabel.Text = "—";
        agreementLabel.FontColor = MUTED;

        referralAgreementLabel.Text = "—";
        referralAgreementLabel.FontColor = MUTED;

        referenceStatus.Text = ...
            "REFERENCE NOT ENTERED";

    end

%% ========================================================================
% GRADE COLOR
% =========================================================================

    function c = gradeColor(index)

        switch index

            case 0

                c = GREEN;

            case 1

                c = BLUE;

            case 2

                c = ORANGE;

            case 3

                c = RED;

            case 4

                c = RED;

            otherwise

                c = MUTED;

        end

    end

%% ========================================================================
% YES / NO
% =========================================================================

    function txt = yesNo(value)

        if value

            txt = "Yes";

        else

            txt = "No";

        end

    end

%% ========================================================================
% WORKFLOW STEP
% =========================================================================

    function makeStep(parent,pos,text,bg)

        p = uipanel(parent);

        p.Position = pos;

        p.BackgroundColor = bg;

        p.BorderType = 'none';

        l = uilabel(p);

        l.Text = text;

        l.FontSize = 11;

        l.FontWeight = 'bold';

        l.FontColor = WHITE;

        l.HorizontalAlignment = 'center';

        l.VerticalAlignment = 'center';

        l.Position = ...
            [0 0 pos(3) pos(4)];

    end

%% ========================================================================
% WORKFLOW TEXT
% =========================================================================

    function makeText(parent,pos,text,txtColor)

        l = uilabel(parent);

        l.Text = text;

        l.FontSize = 9;

        l.FontWeight = 'bold';

        l.FontColor = txtColor;

        l.VerticalAlignment = 'center';

        l.Position = pos;

    end

%% ========================================================================
% WORKFLOW ARROW
% =========================================================================

    function makeArrow(parent,pos)

        l = uilabel(parent);

        l.Text = "→";

        l.FontSize = 17;

        l.FontWeight = 'bold';

        l.FontColor = MUTED;

        l.HorizontalAlignment = 'center';

        l.VerticalAlignment = 'center';

        l.Position = pos;

    end

%% ========================================================================
% RESULT BOX
% =========================================================================

    function createResultBox(parent,pos,titleText)

        p = uipanel(parent);

        p.Position = pos;

        p.BackgroundColor = LIGHTGRAY;

        p.BorderType = 'none';

        l = uilabel(p);

        l.Text = titleText;

        l.FontSize = 8;

        l.FontWeight = 'bold';

        l.FontColor = MUTED;

        l.Position = ...
            [10 pos(4)-28 pos(3)-20 20];

    end

end