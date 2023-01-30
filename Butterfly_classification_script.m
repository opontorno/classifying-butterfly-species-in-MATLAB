%% IMPORT THE DATASET AND BUILDING THE MODEL

path_to_images = "dataset/butterflies";
image_datastore = imageDatastore(path_to_images, "IncludeSubfolders",true,"LabelSource","foldernames");
% or
%image_datastore = load("'.mat' variables/imageDatastore.mat")

%save("imageDatastore", "image_datastore")
%save("network", "net")

%% USING VALIDATION DATASET

[train, validation, test] = splitEachLabel(image_datastore, 0.7, 0.15, 0.15, 'randomized')

%network = load("'.mat' variables/network.mat");
%net = network.net;

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],train);
augimdsValidation = augmentedImageDatastore([224 224 3],validation);

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",20,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",70,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

network_architecture; % save the model structure into lgraph variable
trainingSetup = load("'.mat' variables/params.mat");
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

true_test_labels = test.Labels;
pred_test_labels = classify(net,test);
accuracy_test = mean(true_test_labels == pred_test_labels)

C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)

%% USING THE CROSS VALIDATION

[train_cv, test_cv] = splitEachLabel(image_datastore, 0.7, 0.3, 'randomized');

%network = load("'.mat' variables/network.mat");
%net = network.net;

k=5
cv = cvpartition(train_cv.Labels, 'KFold', k);
network_architecture; % save the model structure into lgraph variable
nets = [];
accuracies = [];

for i=1:k
    idx_train = training(cv,i);
    idx_valid = test(cv,i);
    train = subset(train_cv, idx_train);
    valid = subset(train_cv,idx_valid);
    opts = trainingOptions("sgdm",...
        "ExecutionEnvironment","auto",...
        "InitialLearnRate",0.01,...
        "MaxEpochs",20,...
        "MiniBatchSize",64,...    
        "Shuffle","every-epoch",...
        "ValidationFrequency",70,...
        "Plots","training-progress",...
        "ValidationData",valid);
    [network, traininfo] = trainNetwork(train,lgraph,opts);
    true_valid_labels = valid.Labels;
    pred_valid_labels = classify(network);
    accuracy = mean(true_valid_labels==pred_valid_labels);
    accuracies = [accuracies, accuracy];
    nets = [nets, network];
end

[max_acc,indx_max] = max(accuracies);
cv_net = nets(indx_max);

true_test_labels = test_cv.Labels;
pred_test_labels = classify(cv_net,test_cv);
accuracy_test_cv = mean(true_test_labels == pred_test_labels);

C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)
