[XTrain,TTrain] = japaneseVowelsTrainData;
[XTest,TTest] = japaneseVowelsTestData;

layers = [
    sequenceInputLayer(12)
    lstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(9)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs',10,'Verbose',false);

net = trainNetwork(XTrain,TTrain,layers,options);
YPred = classify(net,XTest);

disp(YPred(1:5))
