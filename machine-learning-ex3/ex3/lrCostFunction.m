function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = rows(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h = sigmoid(X*theta);
for i = 1:m
    J = J - y(i).*log(h(i)) - (1-y(i)).*log(1-h(i));
end
r=0;
for j = 2:n
    r = r + theta(j)^2;
end
r = r .* lambda ./ 2;
J = J + r;
J = J/m;

for j = 1:n
    for i=1:m
        grad(j) = grad(j) + (h(i) - y(i)).*X(i,j);
    end
    grad(j)  = (grad(j))/m;
end
for j = 2:n
    grad(j) = grad(j) + lambda* theta(j)/m;
end
% =============================================================
grad = grad(:);

end
