function allData = convert_ucm(allData)
%CONVERT_UCM Convert UCM to adjacency matrix and store in allData.

OFFSET = 0;%572

for ind = 1:length(allData)
    indOffset = OFFSET + ind;
    fprintf('example %d...\n', ind);
    
    %% get adj list
    ucmMatrix = zeros(size(allData{ind}.adj));
    
    %% get segments
    segments = allData{ind}.segs2;
    
    %% get UCM
    file = sprintf('Data/SB_features/iccv09data/ucm2/iccv09_%d.mat', indOffset);
    load(file);
    
    ucm = ucm2(3:2:end, 3:2:end);
    
    %% get thresholds
    thresholds = sort(unique(ucm(ucm ~= 0)));
    
    [ys,xs] = find(ucm);
    [height, width] = size(ucm);
    
    for i = 1:length(xs)
       x = xs(i);
       y = ys(i);
       
       % try different patterns - and get the threshold/nodes
       if x > 1 && x < width
          if ucm(y, x-1) == 0 && ucm(y, x+1) == 0
              % horizontal
              
              node1 = segments(y, x-1);
              node2 = segments(y, x+1);
              
              threshold = ucm(y, x);
              
              ucmMatrix(node1, node2) = threshold;
              ucmMatrix(node2, node1) = threshold;
          end
       end
       if y > 1 && y < height
           if ucm(y-1, x) == 0 && ucm(y+1, x) == 0
               % vertical
              
              node1 = segments(y-1, x);
              node2 = segments(y+1, x);
              
              threshold = ucm(y, x);
              
              ucmMatrix(node1, node2) = threshold;
              ucmMatrix(node2, node1) = threshold;
           end
       end
       if x > 1 && x < width && y > 1 && y < height
           if ucm(y-1, x-1) == 0 && ucm(y+1, x+1) == 0
               % diagonal
              
              node1 = segments(y-1, x-1);
              node2 = segments(y+1, x+1);
              
              threshold = ucm(y, x);
              
              ucmMatrix(node1, node2) = threshold;
              ucmMatrix(node2, node1) = threshold;
           elseif ucm(y-1, x+1) == 0 && ucm(y+1, x-1) == 0
               % diagonal
              
              node1 = segments(y-1, x+1);
              node2 = segments(y+1, x-1);
              
              threshold = ucm(y, x);
              
              ucmMatrix(node1, node2) = threshold;
              ucmMatrix(node2, node1) = threshold;
           end
       end
    end
    
    %% save to allData
    allData{ind}.ucmAdj = ucmMatrix;
    allData{ind}.ucm = ucm;
end

end