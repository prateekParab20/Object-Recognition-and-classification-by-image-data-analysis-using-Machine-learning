trainingData=imageSet('C:\Users\Prateek\Desktop\images\training','recursive');
[trainingData.Count];
bag= bagOfFeatures(trainingData);
img= read(trainingData(2),randi(trainingData(2).Count));
featureVector = encode(bag,img);
figure;
subplot(3,2,1); imshow(img);
subplot(3,2,2);
bar(featureVector); title('Visual word occurences');xlabel('Visual word Index');ylabel('Frequency of Occurence');


img= read(trainingData(1),randi(trainingData(1).Count));
featureVector = encode(bag,img);
subplot(3,2,3); imshow(img);
subplot(3,2,4);
bar(featureVector); title('Visual word occurences');xlabel('Visual word Index');ylabel('Frequency of Occurence');

%%Train classifier to discriminate between objects
categoryClassifier=trainImageCategoryClassifier(trainingData,bag);

confMatrix=evaluate(categoryClassifier,trainingData);