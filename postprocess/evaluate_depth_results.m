function [ evaluate ] = evaluate_depth_results( preprocessedDir, resultsDir, timeRange, foldRange, searchTypes, splitsName, configFlag )
%EVALUATE_RESULTS Evaluate results in preparation for plotting anytime
%curves.
%
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder and file
%   configFlag:         flag for configuration options

narginchk(4, 7);

if nargin < 7
    configFlag = 1; % not used
end
if nargin < 6
    splitsName = 'splits/Test.txt';
end
if nargin < 5
    searchTypes = [1 2 3 4];
end

%% search types
searchTypesCollection = cell(1, 5);
searchTypesCollection{1} = 'hc';
searchTypesCollection{2} = 'hl';
searchTypesCollection{3} = 'lc';
searchTypesCollection{4} = 'll';
searchTypesCollection{5} = 'rl';

%% test files
testSplitsFile = [preprocessedDir '/' splitsName];
fid = fopen(testSplitsFile, 'r');
list = textscan(fid, '%s');
fclose(fid);
testFiles = list{1,1};

%% depth centers file
depthCentersFile = [preprocessedDir '/' 'depth_centers.txt'];
depthCenters = dlmread(depthCentersFile);

%% prepare output data structure
evaluate = containers.Map;

%% for each search type
for s = searchTypes
    searchType = searchTypesCollection{s};
    fprintf('On search type %s...\n', searchType);
    
    %% prepare data structure
    stat.timeRange = timeRange;
    stat.depthCenters = depthCenters;
    
    stat.deptherrors = zeros(length(foldRange), length(timeRange));
    stat.relativedeptherrors = zeros(length(foldRange), length(timeRange));
    stat.totals = zeros(length(foldRange), length(timeRange));
    
    stat.avgdeptherror = zeros(1, length(timeRange));
    stat.mindeptherror = zeros(1, length(timeRange));
    stat.stddeptherror = zeros(1, length(timeRange));
    
    stat.avgrelativedeptherror = zeros(1, length(timeRange));
    stat.minrelativedeptherror = zeros(1, length(timeRange));
    stat.stdrelativedeptherror = zeros(1, length(timeRange));
    
    %% for each fold
    for fd = 1:length(foldRange)
        fold = foldRange(fd);
        fprintf('\tOn fold %d...\n', fold);

        %% for each file
        for f = 1:length(testFiles)
            fileName = testFiles{f};
            fprintf('\t\tOn file %s...\n', fileName);

%             %% read truth nodes
%             truthNodesPath = [preprocessedDir '/nodes/' fileName '.txt'];
%             [truthLabels, ~] = libsvmread(truthNodesPath);
            
            %% read segments
            segmentsPath = [preprocessedDir '/segments/' fileName '.txt'];
            segments = dlmread(segmentsPath);
            
            %% read truth labeling
            fullTruthPath = [preprocessedDir '/groundtruth_depths/' fileName '.txt'];
            fullTruth = dlmread(fullTruthPath);
            
            %% for each time step
            nodesFileName = sprintf('nodes_%s_test_time%d_fold%d_%s.txt', searchType, timeRange(end), fold, fileName);
            nodesPath = [resultsDir '/results/' nodesFileName];
            inferLabelsMatrix = dlmread(nodesPath);
            inferLabelsMatrix = inferLabelsMatrix';
            
            for t = 1:length(timeRange)
                timeStep = timeRange(t);
                fprintf('\t\t\tOn time step %d...\n', timeStep);

                %% read nodes
                if t > size(inferLabelsMatrix, 2)
                    inferLabels = inferLabelsMatrix(:, end);
                else
                    inferLabels = inferLabelsMatrix(:, t);
                end

                %% read inference on pixel level
                inferPixels = infer_pixels(inferLabels, segments, depthCenters);
                
                stat.deptherrors(fd, t) = stat.deptherrors(fd, t) + sum(sum(double(abs(log10(inferPixels) - log10(fullTruth)))));
                stat.relativedeptherrors(fd, t) = stat.relativedeptherrors(fd, t) + sum(sum(double(abs(inferPixels - fullTruth)./fullTruth)));
                stat.totals(fd, t) = stat.totals(fd, t) + numel(fullTruth);
            end % time range
        end % files
    end % fold

    %% calculate depth errors
    stat.deptherrors = stat.deptherrors ./ stat.totals;
    stat.relativedeptherrors = stat.relativedeptherrors ./ stat.totals;
    
    %% calculate average and standard deviation across folds
    stat.avgdeptherror = mean(stat.deptherrors, 1);
    stat.mindeptherror = max(stat.deptherrors, [], 1);
    stat.stddeptherror = std(stat.deptherrors, 0, 1);
    
    stat.avgrelativedeptherror = mean(stat.relativedeptherrors, 1);
    stat.minrelativedeptherror = max(stat.relativedeptherrors, [], 1);
    stat.stdrelativedeptherror = std(stat.relativedeptherrors, 0, 1);
    
    %% add
    evaluate(searchType) = stat;
end % search type

save([resultsDir '/evaluate.mat'], 'evaluate', '-v7.3');

end

function [inferPixels] = infer_pixels(inferLabels, segments, depthCenters)

inferPixels = zeros(size(segments));
nNodes = length(inferLabels);

for i = 1:nNodes
    temp = segments;
    temp(temp ~= i) = 0;
    temp(temp == i) = depthCenters(inferLabels(i));
    
    inferPixels = inferPixels + temp;
end

end