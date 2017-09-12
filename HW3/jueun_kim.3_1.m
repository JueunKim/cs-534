%----------------------
%3.a Regularization / Linear least_squares regression, LASSO
%----------------------
%% Creating Data set 

dim = 1000; % dimension
n_train = 100;
n_test = 100;
n_realization = 100;

sigma_eta = 0.1;
zero_means = zeros(dim,1);
p_1000 = eye(dim);

beta_init = mvnrnd(zero_means,p_1000);

%set all but 10 weight to zero
beta_init(11:end)=0;
result = zeros(n_realization,4);

%% Least squares & Lasso of Training errors 
for real_idx = 1:n_realization

    % create train set x
    x_train = zeros (n_train,dim);
    for i = 1:n_train
        x_train(i,:) = mvnrnd(zero_means,p_1000);
    end

    % create test set x
    x_test = zeros(n_test,dim);
    for i = 1:n_test
        x_test(i,:) = mvnrnd(zero_means,p_1000);
    end

    % create eta
    eta = mvnrnd(0,sigma_eta^2);

    % create train set y 
    y_train = x_train * beta_init' + eta;

    % create test set y
    y_test = x_test * beta_init';

    %% Use linear regression to fit the model
    b_linearR = LinearModel.fit(x_train,y_train);

    % predict y_train for linear regression
    y_train_pred_linearR = predict(b_linearR,x_train);
    
    % get training error for linear regression
    train_error_linearR = 1/n_train*immse(y_train_pred_linearR,y_train);
    
    % predict y_test for linear regression
    y_test_pred_linearR = predict(b_linearR,x_test);

    % get test error (mse) for linear regression
    test_error_linearR = 1/n_test*immse(y_test_pred_linearR, y_test);
    
    %% Use a lasso fit the model, 5 fold cross validation
    [B_LassoR_all_lambda Stats] = lasso(x_train,y_train,'CV',5);

    % get index of best Lambda 
    best_lambda_index = Stats.IndexMinMSE;
    
    % get the best model with best lambda
    B_LassoR = B_LassoR_all_lambda(:,best_lambda_index);
    
    % predict y_train for linear regression with Lasso
    y_train_pred_LassoR = x_train * B_LassoR;
    % get training error for linear regression with Lasso
    train_error_LassoR = 1/n_train*immse(y_train_pred_LassoR,y_train);
    
    % predict y_test for linear regression with Lasso
    y_test_pred_LassoR = x_test * B_LassoR;

    % get test error (mse) for linear regression with Lasso
    test_error_LassoR = 1/n_test*immse(y_test_pred_LassoR, y_test);

    
    %% save result
    result(real_idx,:) = [train_error_linearR, test_error_linearR, train_error_LassoR, test_error_LassoR];
    
end


%% Plot train errors
figure;
boxplot(result(:,[1 3]),{'Linear Regression','LASSO'})
ylabel('Train error');
title('Train erros');

hold off;
%% Plot test errors
figure;
boxplot(result(:,[2 4]),{'Linear Regression','LASSO'});
ylabel('Test error');
title('Test error');
hold off;
