% Load MNIST dataset from CSV files
train_data = csvread('mnist_train.csv', 1, 0);
test_data = csvread('mnist_test.csv', 1, 0);

% Use only a subset of the data to reduce memory usage
train_data = train_data(1:10000,:);
test_data = test_data(1:2000,:);

% Split dataset into features and labels
X_train = train_data(:,2:end);
y_train = train_data(:,1);
X_test = test_data(:,2:end);
y_test = test_data(:,1);

% Normalize input data
X_train = X_train ./ 255;
X_test = X_test ./ 255;

% Train and test kNN classifier
k = 5;
y_pred = knn_classifier(X_train, y_train, X_test, k);

% Compute confusion matrix and accuracy
C = confusionmat(y_test, y_pred);
accuracy = sum(diag(C)) / sum(C(:));

% Display confusion matrix and accuracy
disp('Confusion Matrix:');
disp(C);
fprintf('Accuracy: %.2f%%\n', 100*accuracy);

function y_pred = knn_classifier(X_train, y_train, X_test, k)
    % Compute pairwise distances between test and training data
    dists = pdist2(X_test, X_train);
    
    % Find k nearest neighbors for each test sample
    [~, indices] = sort(dists, 2);
    k_nearest = indices(:,1:k);
    
    % Predict class labels based on majority vote of nearest neighbors
    y_pred = mode(y_train(k_nearest), 2);
end
