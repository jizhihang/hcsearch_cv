nImages = length(allData);

centers = dlmread([outputPath filesep 'centers.txt']);

depthErrors = [];
relativeDepthErrors = [];
for i = 1:nImages
    initState = allData{i}.initState;
    depths = initState;
    for j = 1:length(initState)
        depths(j) = centers(initState(j));
    end
    
    gtDepths = allData{i}.segDepths;
    
    depthError = abs(log10(gtDepths) - log10(depths));
    relativeDepthError = abs(gtDepths - depths) ./ gtDepths;
    
    depthErrors = [depthErrors; depthError(:)];
    relativeDepthErrors = [relativeDepthErrors; relativeDepthError(:)];
end

avgDepthError = mean(depthErrors);
relativeDepthErrors = mean(relativeDepthErrors);

fprintf('average depth error=%f\n', avgDepthError);
fprintf('average relative depth error=%f\n', relativeDepthErrors);
