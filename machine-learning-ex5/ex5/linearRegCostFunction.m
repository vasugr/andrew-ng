function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;
for i=1:m
    J = J + (h(i) - y(i)).^2;
end
n = size(theta);
for j=2:n
    J =  J + lambda.*(theta(j).^2);
end
J = J/(2*m);

for i=1:n
    for j=1:m
        grad(i) = grad(i) + (h(j) - y(j)).*X(j,i);
    end
    grad(i) = (grad(i))/m;
end
for j = 2:n
    grad(j) = grad(j) + lambda*theta(j)/m;
end
    












% =========================================================================

grad = grad(:);

end
