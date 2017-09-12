close all; clear all; clc;
%----------------------
%1.b. Model complexity
%----------------------

dim = [1, 10, 100]; % dimension
p_dim = {eye(1), eye(10), eye(100)}; %different deviations

n_train = 1000;
n_test = 1000;
n_realization = 100;

sigma_eta =0.01;
result = zeros(n_realization,3);


for dim_index = 1:3
    for real_idx = 1:n_realization
        zero_means = zeros(dim(dim_index),1);
        beta_init = mvnrnd(zero_means,p_dim{dim_index});

        % create train set x
        x_train = zeros(n_train,dim(dim_index));
        for i = 1:n_train
            x_train(i,:) = mvnrnd(zero_means,p_dim{dim_index});
        end

        % create test set x
        x_test = zeros(n_test,dim(dim_index));
        for i = 1:n_test
            x_test(i,:) = mvnrnd(zero_means,p_dim{dim_index});
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
%        mse = (1/n_test)*sqrt(sum(y_test-y_test_pred).^2);
         mse = (1/n_test)*immse(y_test, y_test_pred);
         result(real_idx,dim_index) = mse;
    end
end
 
figure;
boxplot(result(:,:),{'1','10','100'});
ylabel('Optimism');
xlabel('Model Complexity');