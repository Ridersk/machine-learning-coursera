function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


training_size = length(y);
theta_size = length(theta);

z = X * theta;
hypothesis = sigmoid(z);

error = sum((-y .* log(hypothesis)) - (1 - y) .* log(1 - hypothesis));
reg = lambda .* sum(theta(2:end,1).^2);
J = 1/(training_size) * error + 1/(2*training_size) * reg;

for i = 1:theta_size,
  grad(i) = 1/training_size * sum((hypothesis - y) .* X(:, i));
endfor

% Regularization without theta 1
reg_theta = lambda/training_size .* theta(2:end);
grad(2:end) = grad(2:end) .+ reg_theta;

% =============================================================

end
