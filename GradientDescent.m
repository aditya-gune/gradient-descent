%Most current gradient descent as of 10/5/16
%source: http://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re

filename = 'D:\Aditya\Desktop\test p1-16.csv';
rawData = load(filename);
rows = size(rawData,1);
cols = size(rawData,2);

x = rawData(:, 1:cols-1);
y = rawData(:, cols);
normalization = false;
weights = zeros(cols-1,1);
learningRate = 0.01;
reps= 10;

%normalize
if (normalization)
    for i = 1:cols-1
        x(:, i) = (x(:, i) - max(x(:, i))) / (max(x(:, i)) - min(x(:, i)));
    end
    y = (y - max(y)) / (max(y) - min(y));
end

%call gradient to do gradient descent
[weights, costHistory] = gradient(x, y, weights, learningRate, reps);

function [weights, costHistory] = gradient(x, y, weights, learningRate, reps)

lenY = length(y);
lenWts = length(weights);

%vector to hold results of loss function
costHistory = zeros(reps, 1);

%vector to hold theta variable updates after each rep
weightCache = zeros(lenWts,1);

disp(length(weightCache));
disp(length(weights));

for i = 1:reps
    
    %j is result of  the loss function J(w)
    j = ((x * weights) - y)'; %currently spits back y-transpose b/c weights just set x to zero
    
    for k = 1:lenWts
        
        %store recalculated weight into weight cache
        weightCache(k) = weights(k) - learningRate * (1/lenY) * j * x(:, k);
        
        %update the weights
        weights(k) = weightCache(k);
    end
    
    %find cost and record into costhistory
    costHistory(i) = cost(x, y, weights);
    
end

end

function [costVal] = cost(x, y, weights)
    costVal = (x * weights - y)' * (x * weights - y) / (2 * (length(y)));
end
    
