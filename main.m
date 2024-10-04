% Data Preparation
% loads the EMNIST dataset
% imports the images to a double data type
% imports the labels and converts them to ASCII chars
data = load('dataset-letters.mat');
dataset = data.dataset;
im = double(dataset.images)/255;
l = char(dataset.labels + 64);

% creates greyscale 3x4 array of images
% generates random index to show randomly-chosen images
% extracts image using random index
% reshapes image to 28x28
% extracts label using random index and sets as title
% saves figure as PNG file
figure(1), colormap("gray")
random_indices = randperm(size(im, 1));
for i = 1:12
    index = random_indices(i);
    ai = im(index,:);
    ai = reshape(ai,[28,28]);
    subplot(3,4,i), imagesc(ai), axis off, title(num2str(l(index,:)))
end
saveas(gcf, 'sample_data.png');

% calculates size of dataset
% creates randomised order of indexes to shuffle dataset
% determines split point for 50%
% splits dataset into training and testing subsets 
total_samples = size(l,1);
rand_idx = randperm(total_samples);
split_point = round(0.5 * total_samples);

tr_images = im(rand_idx(1:split_point), :);
tr_labels = l(rand_idx(1:split_point));
te_images = im(rand_idx(split_point+1:end), :);
te_labels = l(rand_idx(split_point+1:end));


% 4.1 L2 DISTANCE
% creates empty array to store prediction putput
% loops through each testing image
% creates a replicating matrix so that each training item is compared with the current testing item
% calculates the distance using the L2 measure
% gets the indices of the k smallest L2 values
% gets the labels for the testing data
% calculates correct predictions
% Calculates accuracy and confusion matrix
tepredict_l2 = categorical.empty(size(te_images,1),0);
k_l2 = 1;
tic;
for i = 1:size(te_images,1)
    comp1 = tr_images;
    comp2= repmat(te_images(i,:),[size(tr_images,1),1]);
    l2 = sum((comp1-comp2).^2,2);
    [~,ind] = sort(l2);
    ind = ind(1:k_l2);
    labs = tr_labels(ind);
    tepredict_l2(i,1) = mode(labs);
end
time_l2 = toc;
correct_predictions_l2 = sum(categorical(cellstr(te_labels)) == tepredict_l2);
accuracy_l2 = correct_predictions_l2 /size(te_labels,1);
figure()
confusionchart(categorical(cellstr(te_labels)),tepredict_l2);
title(sprintf('Accuracy=%.2f',accuracy_l2));
saveas(gcf, 'cc_l2.png');



%4.1 L1 DISTANCE
% creates empty array to store prediction putput
% loops through each testing image
% creates a replicating matrix so that each training item is compared with the current testing item
% calculates the distance using the L1 measure
% gets the indices of the k smallest L1 values
% gets the labels for the testing data
% calculates correct predictions
% Calculates accuracy and confusion matrix
tepredict_l1 = categorical.empty(size(te_images,1),0);
k_l1 = 1;
tic;
for i = 1:size(te_images,1)
    comp1 = tr_images;
    comp2= repmat(te_images(i,:),[size(tr_images,1),1]);
    l1 = sum(abs(comp1 - comp2), 2);
    [~,ind] = sort(l1);
    ind = ind(1:k_l1);
    labs = tr_labels(ind);
    tepredict_l1(i,1) = mode(labs);
end
time_l1 = toc;
correct_predictions_l1 = sum(categorical(cellstr(te_labels)) == tepredict_l1);
accuracy_l1 = correct_predictions_l1 /size(te_labels,1);
figure()
confusionchart(categorical(cellstr(te_labels)),tepredict_l1);
title(sprintf('Accuracy=%.2f',accuracy_l1));
saveas(gcf, 'cc_l1.png');




% 4.2 KNM
% trains K-Nearest Neighbour model using training data
% predicts labels for the testing data
% stores time it took to train + test model
% calculates correct predictions 
% calculates accuracy, performance loss and confusion matrix 
tic;
knnmodel = fitcknn(tr_images,tr_labels);
predicted_knn = predict(knnmodel, te_images);
time_knn = toc;
correct_predictions_knn = sum(categorical(cellstr(te_labels)) == predicted_knn);
accuracy_knn = correct_predictions_knn /size(te_labels,1);
knn_resub_err = resubLoss(knnmodel);
figure();
knnmodelCM = confusionchart(te_labels,predicted_knn);
title(sprintf('Accuracy=%.2f',accuracy_l1));
saveas(gcf, 'cc_knn.png');



% 4.2 Decision Tree
% trains Decision Tree model using training data
% predicts labels for the testing data
% stores time it took to train + test model
% calculates correct predictions 
% calculates accuracy, performance loss and confusion matrix
tic;
dtreemodel = fitctree(tr_images,tr_labels);
predicted_dtree = predict(dtreemodel, te_images);
time_dtree = toc;
correct_predictions_dtree = sum(te_labels == predicted_dtree);
accuracy_dtree = correct_predictions_dtree /size(te_labels,1);
dtree_resub_err = resubLoss(dtreemodel);
figure();
dtreemodelCM = confusionchart(te_labels,predicted_dtree);
title(sprintf('Accuracy=%.2f',accuracy_l1));
saveas(gcf, 'cc_dtree.png');


% 4.2 SVM for Multiclass
% trains SVM for Multiclass model using training data
% predicts labels for the testing data
% stores time it took to train + test model
% calculates correct predictions 
% calculates accuracy, performance loss and confusion matrix
tic;
svmmodel = fitctree(tr_images,tr_labels);
predicted_svm = predict(svmmodel, te_images);
time_knn = toc;
correct_predictions_svm = sum(te_labels == predicted_svm);
accuracy_svm = correct_predictions_svm /size(te_labels,1);
svm_resub_err = resubLoss(svmmodel);
figure();
svmmodelCM = confusionchart(te_labels,predicted_svm);
title(sprintf('Accuracy=%.2f',accuracy_l1));
saveas(gcf, 'cc_svm.png');