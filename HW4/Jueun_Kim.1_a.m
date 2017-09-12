sample =[0.25 0.75];
stump = 500;

%% 0.25 train
i = 1
cvpart = cvpartition(Y,'holdout',sample(i));
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

% Train a decision tree ensemble using AdaBoostM1, 500 learning cycles
ClassTreeEns = fitensemble(Xtrain,Ytrain,'AdaBoostM1',stump,'Tree');
figure;
subplot(1,2,1);
title('75% training')
hold on;
plot(loss(ClassTreeEns,Xtrain,Ytrain,'mode','cumulative'),'r');
plot(loss(ClassTreeEns,Xtest,Ytest,'mode','cumulative'));
xlabel('Number of stump');
ylabel('Classification error');
%set(gca,'ylim',[0 1])


%% 0.75 train
i = 2
cvpart = cvpartition(Y,'holdout',sample(i));
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

% Train a decision tree ensemble using AdaBoostM1, 500 learning cycles
ClassTreeEns = fitensemble(Xtrain,Ytrain,'AdaBoostM1',stump,'Tree');
subplot(1,2,2);
title('25% training')
hold on;
plot(loss(ClassTreeEns,Xtrain,Ytrain,'mode','cumulative'),'r');
plot(loss(ClassTreeEns,Xtest,Ytest,'mode','cumulative'));
%set(gca,'ylim',[0 1])
xlabel('Number of stump');
ylabel('Classification error');
hold off;

 

