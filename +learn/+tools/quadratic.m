function y = quadratic(A,B,C,x)
%quadratic 二次多项式函数的实现
%   输入：
%       x 每一列是一个数据点
%   输出：
%       y = 0.5*x'*A*x+B*x+C 二次函数，因为x包含多个数据点，所以使用下面的计算公式
%   
    [~,N] = size(x);
    y = 0.5*sum((x'*A).*x',2)' + B*x + repmat(C,1,N);
end

