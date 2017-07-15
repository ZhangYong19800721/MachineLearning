function y = sample_softmax(x) 
%SAMPLE_SOFTMAX 按照softmax方法抽样，x的每一列是一样例，列向量的每一个元素代表了一个softmax神经元的激活概率
%   
    [row col] = size(x);
    accumulate = zeros(1,col);
    for r = 1:row
        accumulate = accumulate + x(r,:);
        x(r,:) = accumulate;
    end
    
    z = rand(1,col);
    z = repmat(z,row,1);
    y = (x>z);
    k = zeros(size(y));
    k(2:row,:) = y(1:(row-1),:);
    y = y - k;
end

