% path_to_images = "dataset/butterflies";
% image_datastore = imageDatastore(path_to_images, "IncludeSubfolders",true,"LabelSource","foldernames");
% or
image_datastore = load("'.mat' variables/imageDatastore.mat")

[train, validation, test] = splitEachLabel(image_datastore,0.7, 0.15, 0.15, 'randomized');

model; % build and train the googlenet-based model 

true_test_labels = test.Labels;
pred_test_labels = classify(net, test);
accuracy_test = mean(true_test_labels == pred_test_labels)

C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)

save("imageDatastore", "image_datastore")
save("network", "net")
