% Initializations
upper = 10;
train = load('C:\Users\Laurel\Matlab\CS534\src\hw1\train.csv');
test = load('C:\Users\Laurel\Matlab\CS534\src\hw1\test.csv');
numfeatures = size(test,2)-1;
weights = zeros(upper, numfeatures);
costHistory = zeros(upper, upper);

for k = 1:upper
    learningRate = 0.1*k;
    [weights(k,:), costHistory(k,:)] = gradientDescent(learningRate, train, test);
    
    figure;
    plot(1:upper, costHistory(k,:), '-b', 'LineWidth', 2);
    xlabel('Number of iterations');
    ylabel('Cost');
end

figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%source: http://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re
function [weights, costHistory] = gradientDescent(learningRate, train, test)
    rows = size(test,1);
    cols = size(test,2);

    x = test(:, 1:cols-1);
    y = test(:, cols);
    weights = zeros(cols-1,1);
    reps= 10;

    %normalize
    for i = 2:cols-1 %don't normalize dummy feature (std = 0)
        x(:,i) = x(:,i) - mean(x(:,i));  %zero mean: feature - mean(feature)
        x_std = std(x(:,i));
        if x_std ~= 0
            x(:,i) = x(:,i) / x_std;  %unit variance: feature/std(feature) 
        end
    end

    %call gradient to do gradient descent
    [weights, costHistory] = gradient(x, y, weights, learningRate, reps);
end


    function [weights, costHistory] = gradient(x, y, weights, learningRate, reps)

    lenY = length(y);
    lenWts = length(weights);

    %vector to hold results of loss function
    costHistory = zeros(reps, 1);

    %vector to hold theta variable updates after each rep
    weightCache = zeros(lenWts,1);

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

    function [SSE] = cost(x, y, weights)
        %costVal = (x * weights - y)' * (x * weights - y) / (2 * (length(y)));
        SS = sum((x * weights - y)' * (x * weights - y));
        SSE = SS/(2*(length(y)));
    end
    
