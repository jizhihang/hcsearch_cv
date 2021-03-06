function [ marginMeanVals, marginStdVals, marginAccuracies, model ] = margin_performance( allData, evalRange, trainRange, model )
%INITIAL_STATE_PERFORMANCE Summary of this function goes here
%   allData:	data structure containing all preprocessing data
%                   allData{i}.img mxnx3 uint8
%                   allData{i}.labels mxn double
%                   allData{i}.segs2 mxn double
%                   allData{i}.feat2 sxd double
%                   allData{i}.segLabels sx1 double
%                   allData{i}.adj sxs logical
%                   allData{i}.filename string (optional)
%                   allData{i}.segLocations sx2 double (optional)

DISPLAY = 1;

IGNORE_CLASSES = 0;

%% train model
if nargin < 4
    fprintf('Model not specified; training model...\n');
    features = [];
    labels = [];
    
    for i = trainRange
        fprintf('\timage %d\n', i);
        features = [features; allData{i}.feat2];
        labels = [labels; allData{i}.segLabels];
    end
    features = sparse(features);
    fprintf('Training model...\n');
    model = train(labels, features, '-s 7 -c 10');
end

labelOrder = model.Label';
nLabels = length(labelOrder);

%% gather statistics
allMarginCounts = [];
cnt = 1;
marginThresholds = 0:0.1:1;
marginCorrect = zeros(1, length(marginThresholds));
marginTotal = zeros(1, length(marginThresholds));

for i = evalRange
    %% get probabilities
    features = sparse(allData{i}.feat2);
    gtLabels = allData{i}.segLabels;
    segments = allData{i}.segs2;
    [predicted_label, accuracy, probs] = predict(gtLabels, features, model, '-b 1');
    
    %% get the predicted labels ordered by confidence
    [sorted, indices] = sort(probs, 2, 'descend');
    predictedOrderedLabels = labelOrder(indices);
    
    %% compute margin
    margins = abs(sorted - repmat(sorted(:, 1), 1, size(sorted, 2)));
    
    %% for generating rank graphs
    nSegments = size(gtLabels, 1);
    allMarginCounts = [allMarginCounts; zeros(nSegments, length(marginThresholds))];
    for tIndex = 1:length(marginThresholds)
        t = marginThresholds(tIndex);
        
        labelPositionsToKeep = margins <= t;
        restrictedPredict = predictedOrderedLabels .* labelPositionsToKeep + -314*(1-labelPositionsToKeep);
        presence = (restrictedPredict == repmat(gtLabels, 1, size(restrictedPredict, 2)));
        presence = sum(presence, 2);
        segLabels = gtLabels .* presence + (gtLabels+1) .* (1-presence);
        
        counts = sum(labelPositionsToKeep, 2);
        allMarginCounts(cnt:cnt+nSegments-1, tIndex) = counts;
        
        %% ground truth pixel level
        pixelGT = allData{i}.labels;
        pixelGT = pixelGT(:);
        
        %% ground truth segment-level restricted to margin threshold
        segGT = infer_pixels(segLabels, segments);
        segGT = segGT(:);

        %% eliminate IGNORE CLASSES
        for ignoreClass = IGNORE_CLASSES
            ignoreIndices = find(pixelGT == ignoreClass);
            pixelGT(ignoreIndices) = [];
            segGT(ignoreIndices) = [];
        end

        %% compute
        marginCorrect(1, tIndex) = marginCorrect(1, tIndex) + sum(sum(segGT == pixelGT));
        marginTotal(1, tIndex) = marginTotal(1, tIndex) + numel(pixelGT);
    end
    cnt = cnt + nSegments;
end

marginMeanVals = mean(allMarginCounts, 1);
marginStdVals = std(allMarginCounts, 0, 1);
marginAccuracies = marginCorrect ./ marginTotal;

if DISPLAY ~= 0
    figure;
    plot(marginThresholds, marginAccuracies, 's--');
    title('Upper bound pixel accuracy for all margin thresholds');
    xlabel('Threshold');
    ylabel('Upper bound pixel accuracy');

    figure;
    subplot(1,2,1);
    plot(marginThresholds, marginMeanVals, 's--');
    title('Average number of labels kept for all margin thresholds');
    xlabel('Threshold');
    ylabel('Average number of labels kept');

    subplot(1,2,2);
    plot(marginThresholds, marginStdVals, 's--');
    title('Standard deviation of labels kept for all margin thresholds');
    xlabel('Threshold');
    ylabel('Standard deviation of labels kept');
end

end

function [inferPixels] = infer_pixels(inferLabels, segments)

inferPixels = zeros(size(segments));
nNodes = length(inferLabels);

for i = 1:nNodes
    temp = segments;
    temp(temp ~= i) = 0;
    temp(temp == i) = inferLabels(i);
    
    inferPixels = inferPixels + temp;
end

end