function [x,y] = minimize_sgd(F,x0,parameters)
%minimize_sgd 随机梯度下降
%   输入：
%       F 调用F.gradient(x,i)计算目标函数的梯度，调用F.object(x,i)计算目标函数的值，其中i指示minibatch
%       x0 迭代的起始位置
%       parameters 参数集
%   输出：
%       x 最优的参数解
%       y 最小的函数值

    %% 参数设置
    if nargin <= 2 % 没有给出参数
        parameters = [];
        disp('调用minimize_sgd函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'window') % 给出参数但是没有给出window
        parameters.window = 1e+3; 
        disp(sprintf('没有window参数，将使用默认值%f',parameters.window));
    end
    
    if ~isfield(parameters,'max_it') % 给出参数但是没有给出max_it
        parameters.max_it = 1e6;
        disp(sprintf('没有max_it参数，将使用默认值%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'momentum') % 给出参数但是没有给出momentum
        parameters.momentum = 0.9;
        disp(sprintf('没有momentum参数，将使用默认值%f',parameters.momentum));
    end
    
    if ~isfield(parameters,'learn_rate') % 给出参数但是没有给出learn_rate
        parameters.learn_rate = 0.01;
        disp(sprintf('没有learn_rate参数，将使用默认值%f',parameters.learn_rate));
    end
    
    if ~isfield(parameters,'decay') % 给出参数但是没有给出decay
        parameters.decay = 1; % 缺省情况下不降低学习速度
        disp(sprintf('没有decay参数，将使用默认值%f',parameters.decay));
    end
    
    %% 初始化
    inc_x = zeros(size(x0)); % 参数的递增量
    m = parameters.momentum;
    r0 = parameters.learn_rate;
    D = parameters.decay;
    T = parameters.max_it;
    W = parameters.window;
    x1 = x0;  
    y1 = F.object(x1,0); % 计算目标函数值
    
    %% 开始迭代
    z = 100 * y1; k = inf; 
    for it = 1:T
        r  = r0 - (1 - 1/D) * r0 * it / T;
        g1 = F.gradient(x1,it); % 计算梯度
        inc_x = m * inc_x - (1 - m) * r * g1; % 向负梯度方向迭代，并使用动量参数
        x1 = x1 + inc_x; % 更新参数值
        y1 = F.object(x1,it); % 计算目标函数值
        z = (1-1/W)*z + (1/W)*y1;
        disp(sprintf('迭代次数:%d 学习速度:%f 函数均值:%f',it,r,z));
        if mod(it,W) == 0
            if z < k
                k = z;
            else
                break;
            end
        end
    end
    
    %% 返回
    x = x1;
    y = y1;
end

