function [x,y] = maximize_sadam(F,x0,parameters)
%maximize_sadam 梯度法
%   输入：
%       F 调用F.gradient(x)计算目标函数的梯度，调用F.object(x)计算目标函数的值
%       x0 迭代的起始位置
%       parameters 参数
%   输出：
%       x 最优的参数解
%       y 最优的函数值
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_sadam(F,x0,parameters);
    y = -y;
end

