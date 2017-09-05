function [x,y] = maximize_lm(F,x0,parameters)
%maximize_lm 梯度法
%   输入：
%       F 调用F.jacobi(x)计算目标函数的jacobi矩阵，调用F.vector(x)计算目标函数的各个分项值f = [f1,f2,f3,...,fn]
%       目标函数为sum(f.^2)
%       x0 迭代的起始位置
%       parameters.epsilon 当梯度的范数小于epsilon时迭代结束
%       parameters.max_it 最大迭代次数
%   输出：
%       x 最优的参数解
%       y 最小的函数值
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_lm(F,x0,parameters);
    y = -y;
end

