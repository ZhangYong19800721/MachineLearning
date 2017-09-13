function [x1,y1] = maximize_cg(F,x0,parameters)
%minimize_cg 共轭梯度法
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数在x处的值，调用F.gradient(x)计算目标函数在x处的梯度
%   x0 迭代的起始位置
%   parameters 参考minimize_cg的参数设置

    F = learn.optimal.NEGATIVE(F);
    [x1,y1] = learn.optimal.minimize_cg(F,x0,parameters);
    y1 = -y1;
end

