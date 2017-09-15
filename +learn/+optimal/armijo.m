function [ny,nx] = armijo(F,x,g,d,parameters)
%ARMIJO 非精确线搜索的Armijo准则
%   参考 马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社）
%   输入：
%       F 函数对象，F.object(x)计算目标函数在x处的值
%       x 搜索的起始位置
%       g 目标函数在x处的梯度
%       d 搜索方向
%       parameters 参数集
%   输出：
%       ny 最优点的函数值
%       nx 最优点的变量值

    %% 参数设置
    if nargin <= 4 % 没有给出参数的情况
        parameters = [];
        % disp('调用armijo函数时没有给出参数集，将使用默认参数');
    end
    
    if ~isfield(parameters,'beda') % 给出参数但是没有给出beda的情况
        parameters.beda = 0.5; 
        % disp(sprintf('调用armijo函数时参数集中没有beda参数，将使用默认值%f',parameters.beda));
    end
    
    if ~isfield(parameters,'alfa') % 给出参数但是没有给出alfa的情况
        parameters.alfa = 0.2; 
        % disp(sprintf('调用armijo函数时参数集中没有alfa参数，将使用默认值%f',parameters.alfa));
    end
    
    if ~isfield(parameters,'maxs') % 给出参数但是没有给出maxs的情况
        parameters.maxs = 30;
        % disp(sprintf('调用armijo函数时参数集中没有maxs参数，将使用默认值%f',parameters.maxs));
    end

    assert(0 < parameters.beda && parameters.beda <   1);
    assert(0 < parameters.alfa && parameters.alfa < 0.5);
    assert(0 < parameters.maxs);
    
    %%
    d = 1e3 * d; % 搜索方向的值扩大1000倍，相当于可能的最大学习速度为1000
    m = 0; f = F.object(x);
    while m <= parameters.maxs
        nx = x + parameters.beda^m * d;
        ny = F.object(nx);
        if ny <= f + parameters.alfa * parameters.beda^m * g'* d
            break;
        end
        m = m + 1;
    end
end

