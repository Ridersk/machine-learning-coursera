function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
j = zeros(m, 1);

% Acumulate Gradient
D_Theta1 = zeros(size(Theta1));
D_Theta2 = zeros(size(Theta2));

for i = 1:m,
  % set 1 in correspondent indices of labels of y
  y_example = zeros(num_labels, 1);
  y_example(y(i)) = 1;

  z_hidden = [1, X(i, :)] * Theta1';
  hidden_l = sigmoid(z_hidden);
  z_out = [1 , hidden_l] * Theta2';
  out_l = sigmoid(z_out)';
  
  j(i) = 1/m * sum((-y_example .* log(out_l)) - ((1 .- y_example) .* log(1 .- out_l)));
  
  % Errors
  out_err = zeros(num_labels, 1);
  hidden_err = zeros(hidden_layer_size + 1, 1);
  
  out_err = out_l .- y_example;
  hidden_err = Theta2' * out_err;

  % Gradient Acumulate
  D_Theta1(:, 2:end) = D_Theta1(:, 2:end) + hidden_err(2:end) * X(i, :);
  
  D_Theta2(:, 2:end) = D_Theta2(:, 2:end) + out_err * hidden_l;
endfor

Theta1_grad = D_Theta1 / m;
Theta2_grad = D_Theta2 / m;
  
% -------------------------------------------------------------
# J Total
J = sum(j);


# Regularization
r_theta1 = sum(sum(Theta1(:, 2:end) .^2, 2));
r_theta2 = sum(sum(Theta2(:, 2:end) .^2, 2));
regularization = lambda/(2 * m) * (r_theta1 + r_theta2);

J = J + regularization;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
