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
X = [ones(m, 1) X];
% First hidden layer.
z2 = X * Theta1';
a2 = sigmoid(z2);
% Add ones to this hidden layer.
a2 = [ones(m, 1) a2];
% Output layer.
z3 = a2 * Theta2';
h_thetas = sigmoid(z3);
% Compute J without regularization.
y_matrix = eye(num_labels)(y, :);
cm = log(1-h_thetas).*(y_matrix-1) - log(h_thetas).*y_matrix;
J = sum(cm(:)) / m;  
% Compute J with regularization.
th1 = Theta1(:, 2:end);
reg = sumsq(th1(:));
th2 = Theta2(:, 2:end);
reg += sumsq(th2(:));
J += reg * lambda / 2 / m;
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
delta3 = h_thetas - y_matrix;
delta2 = delta3*th2 .* sigmoidGradient(z2);
Theta2_grad = delta3' * a2 / m;
Theta1_grad = delta2' * X / m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Add zeros columns to th1 & th2 so that the bias columns are not regularized.
th2 = [zeros(size(Theta2, 1), 1) th2];
Theta2_grad += th2 * lambda / m;
th1 = [zeros(size(Theta1, 1), 1) th1];
Theta1_grad += th1 * lambda / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
