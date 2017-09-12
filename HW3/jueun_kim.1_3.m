close all; clear all; clc;
%----------------------
%1.c. Sample size
%----------------------
dim = 10; % dimension
n_train = [1000, 100, 50];
n_test = 1000;
n_realization = 100;

sigma_eta =0.01;
zero_means = zeros(dim,1);
p_10 = eye(dim);

beta_init = mvnrnd(zero_means,p_10);

result = zeros(n_realization,length(sigma_eta));

for index = 1:3
    for real_idx = 1:n_realization

        % create train set x
        x_train = zeros(n_train(index),dim);
        for i = 1:n_train(index)
            x_train(i,:) = mvnrnd(zero_means,p_10);
        end

        % create test set x
        x_test = zeros(n_test,dim);
        for i = 1:n_test
            x_test(i,:) = mvnrnd(zero_means,p_10);
        end

        % create eta
        eta = mvnrnd(0,sigma_eta^2);

        % create train set y 
        y_train = x_train * beta_init' + eta;
        
        % create test set y
        y_test = x_test * beta_init';
        
        % train b_model 
        b_model = LinearModel.fit(x_train,y_train);
        
        % predict y_test
        y_test_pred = predict(b_model,x_test);

        % get mse for y_test and y_test_pred
        err = (1/n_test)*immse(y_test, y_test_pred);
        result(real_idx,index) = err;
    end
end

figure;
boxplot(result(:,:),{'1000','100','50'});
ylabel('Optimism');
xlabel('Sample Size');
title('Sample Size');