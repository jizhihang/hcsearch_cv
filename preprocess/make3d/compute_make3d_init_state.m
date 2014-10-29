%% output file here
outputPath = 'Data/Make3D_Pre_Test';
outputPath = normalize_file_sep(outputPath);

%% constants
DUMMY_VALUE = 1;

EXTERNAL_PATH = 'external';
LIBLINEAR_PATH = [EXTERNAL_PATH filesep 'liblinear'];

%% create folder
if ~exist([outputPath filesep 'initstate' filesep], 'dir')
    mkdir([outputPath filesep 'initstate' filesep]);
end

%% get features
nFiles = length(allData);
trainNodeLabels = [];
trainNodeFeatures = [];
for i = 1:nFiles 
    isTrainingImage = isfield(allData{i}, 'segLabels');
    
    if isTrainingImage
        trainNodeLabels = [trainNodeLabels; allData{i}.segDepths];
        trainNodeFeatures = [trainNodeFeatures; allData{i}.feat2];
    end
end

%% create initial classifier training file
INITFUNC_TRAINING_FILE = 'initfunc_regression_training.txt';
libsvmwrite([outputPath filesep INITFUNC_TRAINING_FILE], trainNodeLabels, sparse(trainNodeFeatures));

%% train initial prediction classifier on the training file just generated
INITFUNC_MODEL_FILE = 'initfunc_regression_model.txt';
fprintf('Training initial regression model...\n');
if ispc
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'windows' filesep 'train'];
elseif isunix
    LIBLINEAR_TRAIN = [LIBLINEAR_PATH filesep 'train'];
end

LIBLINEAR_TRAIN_CMD = [LIBLINEAR_TRAIN ' -s 12 -c 10 ' ...
    outputPath filesep INITFUNC_TRAINING_FILE ' ' ...
    outputPath filesep INITFUNC_MODEL_FILE];

if ispc
    dos(LIBLINEAR_TRAIN_CMD);
elseif isunix
    unix(LIBLINEAR_TRAIN_CMD);
end
initStateModel = train(trainNodeLabels, sparse(trainNodeFeatures), '-s 12 -c 10');

%% generate the initial prediction files
centers = dlmread([outputPath filesep 'centers.txt']);
for i = 1:nFiles
    fprintf('Predicting example %d...\n', i-1);
    
    filename = sprintf('%d', i-1);
    if isfield(allData{i}, 'filename');
        filename = allData{i}.filename;
    else
        allData{i}.filename = filename;
    end
    
    initPredFile = sprintf('%s.txt', filename);
    nodesFile = sprintf('%s.txt', filename);
    
    if ispc
        LIBLINEAR_PREDICT = [LIBLINEAR_PATH filesep 'windows' filesep 'predict'];
    elseif isunix
        LIBLINEAR_PREDICT = [LIBLINEAR_PATH filesep 'predict'];
    end
    
    LIBLINEAR_PREDICT_CMD = [LIBLINEAR_PREDICT ' ' ...
        outputPath filesep 'nodes' filesep nodesFile ' ' ...
        outputPath filesep INITFUNC_MODEL_FILE ' ' ...
        outputPath filesep 'initstate' filesep initPredFile];
    
    if ispc
        dos(LIBLINEAR_PREDICT_CMD);
    elseif isunix
        unix(LIBLINEAR_PREDICT_CMD);
    end
    
    [initStateLabels, ~, ~] = predict(DUMMY_VALUE*ones(size(allData{i}.feat2, 1), 1), sparse(allData{i}.feat2), initStateModel);
    allData{i}.initDepths = initStateLabels;
    allData{i}.initState = initStateLabels;
    
    for row = 1:length(allData{i}.initState)
        [~, allData{i}.initState(row)] = min(vl_alldist(allData{i}.initDepths(row), centers'));
    end
end
