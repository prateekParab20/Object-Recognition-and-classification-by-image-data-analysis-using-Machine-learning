%% Conversion of images to grayscale for 1st training example
gscale();
%% Conversion of images to grayscale for 1st training example
gscale();
%% Load Data
trainingData=imageSet('E:\wallpaper','recursive');

%% Display classnames and counts
tbl=trainingData(1,1).Count;
[test_set_a,training_set_a] = partition(trainingData(1,1),194);
aeroTrain=training_set_a.Count;
aeroTest=test_set_a.Count;

tbl2=trainingData(1,2).Count;
[test_set_c,training_set_c] = partition(trainingData(1,2),74);
carTrain=training_set_c.Count;
carTest=test_set_c.Count;


 %% Display Grayscale images
%  for i=1:aeroTrain
% [m,n]=size('training_set_a.ImageLocation');
% imtool(a);
     %% Combining Image Training sets
A=training_set_a;
[m, n]=size(A);
mdup=1; ndup=2;

CombiTrain=repmat(A,[1 1 mdup ndup]);
CombiTrain=permute(CombiTrain,[3 1 4 2]);
CombiTrain=reshape(CombiTrain,m*mdup,n*ndup);
CombiTrain(1,2)=training_set_c(1,1);

 


%% Create Visual Vocabulary 
bag= bagOfFeatures(CombiTrain,'VocabularySize',500,'PointSelection','Detector');
DataScene = double(encode(bag, CombiTrain));

%  bagCar= bagOfFeatures(training_set_c,'VocabularySize',1200,'PointSelection','Detector');
%  carDataScene = double(encode(bagCar, training_set_c));

%% Visualize Feature Vectors 
img_a = read(CombiTrain(1), randi(CombiTrain(1).Count));
featureVector = encode(bag, img_a);

subplot(4,2,1); 
imshow(img_a);
subplot(4,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(CombiTrain(2), randi(CombiTrain(2).Count));
featureVector = encode(bag, img);
subplot(4,2,3); 
imshow(img);
subplot(4,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');


%% Create a Table using the encoded features
SceneImageData = array2table(DataScene);
sceneType = categorical(repelem({CombiTrain.Description}', [CombiTrain.Count], 1));
SceneImageData.sceneType = sceneType;
%% Use the new features to train a model and assess its performance using 
%      classificationLearner
% x=trainedClassifier.ClassificationSVM.X;
% y=trainedClassifier.ClassificationSVM.Y;
%  SVMModel = fitcsvm(x,y);

%% Combining test sets
A=test_set_a;
[m, n]=size(A);
mdup=1; ndup=2;

CombiTest=repmat(A,[1 1 mdup ndup]);
CombiTest=permute(CombiTest,[3 1 4 2]);
CombiTest=reshape(CombiTest,m*mdup,n*ndup);
CombiTest(1,2)=test_set_c(1,1);
 %% Testing data
testSceneData = double(encode(bag,CombiTest));
testSceneData = array2table(testSceneData,'VariableNames',trainedClassifier.RequiredVariables);
actualSceneType = categorical(repelem({CombiTest.Description}', [CombiTest.Count], 1));

predictedOutcome = trainedClassifier.predictFcn(testSceneData);

correctPredictions = (predictedOutcome == actualSceneType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome)*100;

 %% Visualize how the classifier works
ii = randi(size(CombiTest,2));
jj = randi(CombiTest(ii).Count);
img = read(CombiTest(ii),jj);

imshow(img);

% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));

% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationSVM,imagefeatures);
% Display the string label for img
if strcmp(char(bestGuess),CombiTest(ii).Description)
    titleColor = [0 0.8 0];
else
    titleColor = 'r';
end
title(sprintf('Prediction: %s; Actual: %s',...
    char(bestGuess),CombiTest(ii).Description),...
    'color',titleColor)