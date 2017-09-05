function [x,y] = maximize_g(F,x0,parameters)
%maximize_g 梯度法
%   输入：
%       F 调用F.gradient(x)计算目标函数的梯度，调用F.object(x)计算目标函数的值
%       x0 迭代的起始位置
%       parameters.learn_rate 学习速度
%       parameters.momentum 加速动量
%       parameters.epsilon 当梯度的范数小于epsilon时迭代结束
%       parameters.max_it 最大迭代次数
%   输出：
%       x 最优的参数解
%       y 最小的函数值
    
    F = learn.optimal.NEGATIVE(F);
    [x,y] = learn.optimal.minimize_g(F,x0,parameters);
    y = -y;
end

