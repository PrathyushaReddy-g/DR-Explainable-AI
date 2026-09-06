%% =========================================================
% BENCHMARK COMPARISON
% Explainable AI for Diabetic Retinopathy Screening
% ==========================================================

clc;
clear;
close all;

%% =========================================================
% 1. CREATE RESULTS FOLDER
% ==========================================================

if ~exist('results','dir')
    mkdir('results');
end

%% =========================================================
% 2. MODEL BENCHMARK DATA
% ==========================================================

benchmarks = {
    'Our ResNet-18',                  'Current Test Set', 78.96, 80.29, 95.63;
    'ResNet-18 [Literature]',         'APTOS-2019',      97.80, 94.57, NaN;
    'CNN [Literature]',               'APTOS-2019',      94.60, 86.00, 96.00;
    'ResNet-50 [Literature]',         'APTOS-2019',      94.21, 94.07, 94.34;
    'EfficientNet-B0 [Literature]',  'APTOS-2019',      95.73, 95.38, 95.88;
    'ViT [Literature]',               'APTOS-2019',      96.85, 96.44, 97.07
    };

%% =========================================================
% 3. CONVERT DATA INTO TABLE
% ==========================================================

T = cell2table(benchmarks, ...
    'VariableNames', { ...
    'Model', ...
    'Dataset', ...
    'Accuracy', ...
    'Sensitivity', ...
    'Specificity'});

%% =========================================================
% 4. DISPLAY TABLE
% ==========================================================

disp(' ');
disp('====================================================');
disp('DIABETIC RETINOPATHY BENCHMARK COMPARISON');
disp('====================================================');
disp(' ');

disp(T);

%% =========================================================
% 5. SAVE TABLE
% ==========================================================

writetable(T, ...
    'results\benchmark_comparison.csv');

disp(' ');
disp('Benchmark table saved successfully.');
disp('Location: results\benchmark_comparison.csv');

%% =========================================================
% 6. PREPARE DATA FOR CHART
% ==========================================================

numModels = height(T);

accuracy = T.Accuracy;
sensitivity = T.Sensitivity;
specificity = T.Specificity;

%% =========================================================
% 7. CREATE PROFESSIONAL HORIZONTAL BAR CHART
% ==========================================================

figure( ...
    'Name','DR Benchmark Comparison', ...
    'NumberTitle','off', ...
    'Position',[100 100 1250 700]);

%% Numeric positions
% Using numeric positions prevents MATLAB from
% automatically reordering the model names.

y = 1:numModels;

values = [ ...
    accuracy, ...
    sensitivity, ...
    specificity ...
    ];

%% Create horizontal bars

barh(y, values);

%% =========================================================
% 8. KEEP OUR MODEL AT THE TOP
% ==========================================================

set(gca, 'YDir', 'reverse');

%% =========================================================
% 9. MODEL LABELS
% ==========================================================

yticks(y);

yticklabels(T.Model);

%% =========================================================
% 10. X-AXIS
% ==========================================================

xlabel( ...
    'Performance (%)', ...
    'FontSize',13, ...
    'FontWeight','bold');

xlim([0 105]);

xticks(0:10:100);

%% =========================================================
% 11. Y-AXIS
% ==========================================================

ylabel( ...
    'Model', ...
    'FontSize',13, ...
    'FontWeight','bold');

%% =========================================================
% 12. TITLE
% ==========================================================

title( ...
    'Diabetic Retinopathy Model Benchmark Comparison', ...
    'FontSize',16, ...
    'FontWeight','bold');

%% =========================================================
% 13. GRID
% ==========================================================

grid on;

set(gca, ...
    'FontSize',11, ...
    'LineWidth',1);

%% =========================================================
% 14. LEGEND
% ==========================================================

legend( ...
    'Accuracy', ...
    'Sensitivity', ...
    'Specificity', ...
    'Location','southoutside', ...
    'Orientation','horizontal');

%% =========================================================
% 15. ADD VALUE LABELS
% ==========================================================

hold on;

for i = 1:numModels

    % ---------------------------------------------
    % Accuracy
    % ---------------------------------------------

    if ~isnan(accuracy(i))

        text( ...
            accuracy(i) + 0.8, ...
            i - 0.25, ...
            sprintf('%.2f%%', accuracy(i)), ...
            'FontSize',9, ...
            'VerticalAlignment','middle');

    end

    % ---------------------------------------------
    % Sensitivity
    % ---------------------------------------------

    if ~isnan(sensitivity(i))

        text( ...
            sensitivity(i) + 0.8, ...
            i, ...
            sprintf('%.2f%%', sensitivity(i)), ...
            'FontSize',9, ...
            'VerticalAlignment','middle');

    end

    % ---------------------------------------------
    % Specificity
    % ---------------------------------------------

    if ~isnan(specificity(i))

        text( ...
            specificity(i) + 0.8, ...
            i + 0.25, ...
            sprintf('%.2f%%', specificity(i)), ...
            'FontSize',9, ...
            'VerticalAlignment','middle');

    end

end

hold off;

%% =========================================================
% 16. IMPROVE FIGURE SIZE
% ==========================================================

set(gcf, ...
    'Color','white');

%% =========================================================
% 17. SAVE STANDARD PNG
% ==========================================================

saveas( ...
    gcf, ...
    'results\benchmark_comparison.png');

%% =========================================================
% 18. SAVE HIGH-RESOLUTION PNG
% ==========================================================

exportgraphics( ...
    gcf, ...
    'results\DR_Benchmark_Comparison_Professional.png', ...
    'Resolution',300);

%% =========================================================
% 19. FINAL OUTPUT
% ==========================================================

disp(' ');
disp('====================================================');
disp('BENCHMARK COMPARISON COMPLETED SUCCESSFULLY');
disp('====================================================');
disp(' ');
disp('Generated files:');
disp(' ');
disp('1. results\benchmark_comparison.csv');
disp('2. results\benchmark_comparison.png');
disp('3. results\DR_Benchmark_Comparison_Professional.png');
disp(' ');
disp('The benchmark chart is ready.');
disp(' ');