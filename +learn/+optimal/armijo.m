function [lamda,nf,nx] = armijo(F,x,g,d,parameters)
%ARMIJO 非精确线搜索的Armijo准则
%   参考 马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社）
%   输入：
%       F 函数对象，F.object(x)计算目标函数在x处的值
%       x 搜索的其实位置
%       g 目标函数在x处的梯度
%       d 搜索方向
%       parameters.armijo.beda 值必须在(0,  1)之间，典型值0.5
%       parameters.armijo.alfa 值必须在(0,0.5)之间，典型值0.2
%       parameters.armijo.maxs 最大搜索步数(正整数)，典型值30
%
%   输出：
%       lamda 搜索步长
    if nargin <= 4
        parameters.armijo.beda = 0.5;
        parameters.armijo.alfa = 0.2;
        parameters.armijo.maxs = 30;
    end

    assert(0 < parameters.armijo.beda && parameters.armijo.beda <   1);
    assert(0 < parameters.armijo.alfa && parameters.armijo.alfa < 0.5);
    assert(0 < parameters.armijo.maxs);
    d = 1e3 * d; % 搜索方向的值扩大1000倍，相当于可能的最大学习速度为1000
    m = 0; f = F.object(x);
    while m <= parameters.armijo.maxs
        nx = x + parameters.armijo.beda^m * d;
        nf = F.object(nx);
        lamda = parameters.armijo.beda^m;
        if nf <= f + parameters.armijo.alfa * lamda * g'* d
            break;
        end
        m = m + 1;
    end
end

