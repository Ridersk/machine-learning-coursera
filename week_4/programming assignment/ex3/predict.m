function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% Layer1 (Input Layer) = X values.
% Note: examples was ordered from line to columns
% transpose X from 5000x400 to 400x5000
% out1: 401x5000 (layer_units X num_examples)
out1 = transpose([ones(m, 1) X]);

% Layer2 (Hidden Layer)
% out2: 26x5000 (layer_units X num_examples)
layer2_units = size(Theta1, 1);
out2 = [ones(1, m); sigmoid(Theta1 * out1)];

% Layer3 (Output Layer)
% out3: 10x5000
out3 = sigmoid(Theta2 * out2);

[values, p] = max(out3, [], 1);

% Transpose 1x5000 to 5000x1 (examples X labels)
p = transpose(p);

% =========================================================================


end
