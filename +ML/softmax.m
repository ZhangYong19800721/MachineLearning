function y = softmax(x)
%SOFTMAX 计算softmax神经元的激活概率
%   
    N = size(x,1);
    y = exp(x);
    y = y ./ repmat(sum(y),N,1);
end

