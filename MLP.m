% Loading the MNIST dataset from CSV files
train_data = csvread('mnist_train.csv', 1, 0);
test_data = csvread('mnist_test.csv', 1, 0);

% Splitting the dataset into features and labels
X_train = train_data(:, 2:end);
y_train = train_data(:, 1);
X_test = test_data(:, 2:end);
y_test = test_data(:, 1);

% Normalizing the input data
X_train = X_train./255;
X_test = X_test./255;

% Converting the labels to one-hot encoding
y_train_onehot = zeros(length(y_train), 10);
for i = 1:length(y_train)
    y_train_onehot(i, y_train(i)+1) = 1;
end
y_train = y_train_onehot;

y_test_onehot = zeros(length(y_test), 10);
for i = 1:length(y_test)
    y_test_onehot(i, y_test(i)+1) = 1;
end
y_test = y_test_onehot;

% Initializing the weights and biases
input_dim = size(X_train, 2);
hidden_dim = 10000;
output_dim = 10;
learning_rate = 0.003;
num_iterations = 2;
batch_size = 64;
num_batches = floor(size(X_train, 1)/batch_size);

W1 = randn(input_dim, hidden_dim)./sqrt(input_dim);
b1 = zeros(1, hidden_dim);
W2 = randn(hidden_dim, output_dim)./sqrt(hidden_dim);
b2 = zeros(1, output_dim);

% Training the model
for i = 1:num_iterations
    % Shuffle the training data
    idx_shuffle = randperm(size(X_train, 1));
    X_train = X_train(idx_shuffle, :);
    y_train = y_train(idx_shuffle, :);
    
    for j = 1:num_batches
        % Select a minibatch
        start_idx = (j-1)*batch_size+1;
        end_idx = j*batch_size;
        X_batch = X_train(start_idx:end_idx, :);
        y_batch = y_train(start_idx:end_idx, :);
        
        % Forward pass
        z1 = X_batch*W1 + b1;
        a1 = sigmoid(z1);
        z2 = a1*W2 + b2;
        y_pred = softmax(z2);
        
        % Backward pass
        delta3 = y_pred - y_batch;
        delta2 = delta3*W2'.*(a1.*(1-a1));
                
        % Update weights and biases
        W2 = W2 - learning_rate*a1'*delta3;
        b2 = b2 - learning_rate*sum(delta3, 1);
        W1 = W1 - learning_rate*X_batch'*delta2;
        b1 = b1 - learning_rate*sum(delta2, 1);
    end
end

% Testing the model
z1 = X_test*W1 + b1;
a1 = sigmoid(z1);
z2 = a1*W2 + b2;
y_pred = softmax(z2);
[~, y_pred] = max(y_pred, [], 2);
[~, y_test] = max(y_test, [], 2);
accuracy = sum(y_pred == y_test)/length(y_test);

% Computing the confusion matrix
C = confusionmat(y_test, y_pred);
disp("Confusion Matrix:");
disp(C);

% Plotting the confusion matrix
figure;
imagesc(C);
title("Confusion Matrix");
xlabel("Predicted Labels");
ylabel("True Labels");
colorbar;


function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end