% Load GoogLeNet
net1 = googlenet;
disp('GoogLeNet Model:');
disp(net1);

% Load MobileNetV2
net2 = mobilenetv2;
disp('MobileNetV2 Model:');
disp(net2);

% Create and train RNN
[X,T] = simpleseries_dataset;

net3 = layrecnet(1:2,10);

[Xs,Xi,Ai,Ts] = preparets(net3,X,T);

net3 = train(net3,Xs,Ts,Xi,Ai);

% Display RNN architecture
view(net3);
