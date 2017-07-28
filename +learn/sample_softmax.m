function y = sample_softmax(x) 
    %SAMPLE_SOFTMAX softmax抽样 
    %   x的每一列是一样例，列向量的每一个元素代表了一个softmax神经元的激活概率
    
    [M,N] = size(x);           % 样本值的个数
    y = zeros(size(x));        % y初始化为全0
    P = cumsum(x);             % 累积概率
    R = repmat(rand(1,N),M,1); % 产生随机变量
    D = 1 + sum(P<R); 
    I = sub2ind(size(x),D,1:N); 
    y(I) = 1;
end

