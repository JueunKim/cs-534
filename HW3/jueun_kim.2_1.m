%----------------------
%2.a Feature selection the wrong way
%----------------------
dim = 5000; % dimension
n_train = 100;
n_test = 100;
n_realization = 100;

p_5000 = eye(dim);
zero_means = zeros(dim,1);
result = zeros(n_realization,1);

for real_idx = 1:n_realization

    %Generate train set x, test set y
    x_train = zeros(100,dim);
    x_test = zeros(n_test,dim); 
   
    for idx = 1:n_train 
        x_train(idx,:) = mvnrnd(zero_means,p_5000);
        x_test(idx, : ) = mvnrnd(zero_means,p_5000);    
    end
    
    %Generate random train y.
    y_train = 0+(1-0).*rand(100,1);
    
    %bernoulli(0.5)
    for idx = 1:100
        if y_train(idx,:) > 0.5    
           y_train(idx,:) = 1;
        else y_train(idx,:) = 0;
        end
    end

    %Generate random test y.
    y_test = 0+(1-0).*rand(100,1);
    for idx = 1:100
        if y_test(idx,:) > 0.5    
           y_test(idx,:) = 1;
        else y_test(idx,:) = 0;
        end
    end
    
    %combine training and testing sample
    combine_set = [x_test x_train];
    %evaluate correlation between each individual x_i & Y
    V= zeros(dim,1);
    for idx = 1:dim
        A = corrcoef(combine_set(:,idx), y_train);
        V(idx) = abs(A(1,2));
    end
    
    % Select the 10 most highly correlated features.
    [SortedValue, SortedIndex] = sort(V(:), 'descend');
    x_train_selected = SortedIndex(1:10);

    mdl = fitcknn(x_train,y_train, 'NumNeighbors', 1);

    y_test_pred = predict(mdl,x_test);

    err = (1/n_test)*immse(y_test, y_test_pred);
    result(real_idx,1) = err;
end


figure;
boxplot(result(:,:),{'incorrect'});
ylabel('');
xlabel('Testing error');
title(['Feature selection the wrong way, err = ',num2str(err)]);