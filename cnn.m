% Load the MNIST dataset from CSV files
train_data = readmatrix('mnist_train.csv', 'NumHeaderLines', 1);
test_data = readmatrix('mnist_test.csv', 'NumHeaderLines', 1);

% Remove the first column (labels)
X_train = train_data(:, 2:end);
X_test = test_data(:, 2:end);

% The first column corresponds to labels
y_train = categorical(train_data(:, 1));
y_test = categorical(test_data(:, 1));

% Reshape the data to [Height Width Channels Observations]
X_train = reshape(X_train', [28, 28, 1, size(X_train, 1)]);
X_test = reshape(X_test', [28, 28, 1, size(X_test, 1)]);

% Normalize the data
X_train = X_train / 255;
X_test = X_test / 255;

% Define the LeNet-5 architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(5, 20)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(5, 50)
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(500)
    reluLayer()
    
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()
];

% Specify the training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128);

% Train the CNN using the specified options
net = trainNetwork(X_train, y_train, layers, options);

% Evaluate the trained CNN on the test set
y_pred = classify(net, X_test);
accuracy = sum(y_pred == y_test) / numel(y_test);
disp(['Accuracy: ', num2str(accuracy * 100), '%'])

% Plot confusion matrix
figure
confusionchart(y_test, y_pred);
