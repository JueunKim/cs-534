% 2.b Built-in logistic regression
load('HW4.mat')
Data = [X Y];
[Xtrain, valInd, Xtest] = dividerand(Data(:,1:30)',.75,0,.25);
[Ytrain, valInd, Ytest] = dividerand(Data(:,31)',.75,0,.25);

Xtrain = Xtrain';
Xtest =Xtest';
Ytrain = Ytrain';
Ytest = Ytest';

Ytrain= Ytrain+2;

% do logistic regression
weight = mnrfit(Xtrain,Ytrain,'Model','nominal');

%% add bias space to the xtest :: first row of weight is intercept (bias)
%xtrain_with_bias =[ones(size(Xtrain,1),1) Xtrain];
xtest_with_bias = [ones(size(Xtest,1),1) Xtest];

%% predict Xtest
y_test_pred = xtest_with_bias * weight(:,1) ;
%y_train_pred = xtrain_with_bias * weight(:,1) ;


for i=1:length(y_test_pred)
    if y_test_pred(i,:) > 0 
        y_test_pred(i,:) = 3;
    end
    if y_test_pred(i,:) <0 
        y_test_pred(i,:) = 1;
    end
end

%%
%train_mse = 1/length(Xtrain)*sqrt(sum(y_train_pred - Ytrain).^2);
test_mse = 1/length(Xtest)*sqrt(sum((y_test_pred - Ytest).^2));
x = 1:1000;
figure;
plot(x,test_mse(1,:),'g*');
hold on;   
title('mnrfit Logistic Regression');
xlabel('Iteration');
ylabel('Misclassification Rate'); 

