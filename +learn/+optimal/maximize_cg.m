function [x1,y1] = maximize_cg(F,x0,parameters)
%minimize_cg 共轭梯度法
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数在x处的值，调用F.gradient(x)计算目标函数在x处的梯度
%   x0 迭代的起始位置
%   parameters.epsilon 当梯度模小于epsilon时停止迭代
%   parameters.alfa 线性搜索区间倍数
%   parameters.beda 线性搜索的停止条件
%   parameters.max_it 最大迭代次数
%   parameters.reset 重置条件

    F = learn.optimal.NEGATIVE(F);
    [x1,y1] = learn.optimal.minimize_cg(F,x0,parameters);
    y1 = -y1;
end

