function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
predictions = (Theta * X')';
error_matrix = ((predictions - Y).^2);
error_matrix = error_matrix(R == 1); % regard only predictions in which the user rated
reg = lambda*sum(sum(Theta.^2))/2 + lambda*sum(sum(X.^2))/2; % regularization
J = sum(sum(error_matrix))/2 + reg;

# Consider only people that rated the movie to adjust the matrix X (moviesXfeatures)
for i = 1:num_movies,
  idx = find(R(i, :)==1);
  Theta_temp = Theta(idx, :);
  Y_temp = Y(i, idx);
  predictions = X(i,:)*Theta_temp';
  reg = lambda*X(i,:);
  X_grad(i, :) = (predictions - Y_temp)*Theta_temp + reg;
endfor

# Consider only movies that the user rated to adjust the matrix Theta (usersXfeatures)
for i = 1:num_users,
  idx = find(R(:, i) == 1);
  X_temp = X(idx, :);
  Y_temp = Y(idx, i);
  predictions = Theta(i, :)*X_temp';
  reg = lambda*Theta(i,:);
  Theta_grad(i, :) = (predictions - Y_temp')*X_temp + reg;
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
