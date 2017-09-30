function [x,y] = minimize_sadam(F,x0,parameters)
%minimize_adam Stochastic ADAM 随机梯度下降
%   参考文献“ADAM:A Method For Stochastic Optimization”,2014
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
        disp('调用minimize_adam函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon
        parameters.epsilon = 1e-8; 
        disp(sprintf('没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'window') % 给出参数但是没有给出window
        parameters.window = 1e+3; 
        disp(sprintf('没有window参数，将使用默认值%f',parameters.window));
    end
    
    if ~isfield(parameters,'max_it') % 给出参数但是没有给出max_it
        parameters.max_it = 1e6;
        disp(sprintf('没有max_it参数，将使用默认值%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'learn_rate') % 给出参数但是没有给出learn_rate
        parameters.learn_rate = 1e-3;
        disp(sprintf('没有learn_rate参数，将使用默认值%f',parameters.learn_rate));
    end
    
    if ~isfield(parameters,'beda1') 
        parameters.beda1 = 0.9;
        disp(sprintf('没有beda1参数，将使用默认值%f',parameters.beda1));
    end
    
    if ~isfield(parameters,'beda2') 
        parameters.beda2 = 0.999;
        disp(sprintf('没有beda2参数，将使用默认值%f',parameters.beda2));
    end
    
    if ~isfield(parameters,'decay') % 给出参数但是没有给出decay
        parameters.decay = 0; % 缺省情况下不降低学习速度
        disp(sprintf('没有decay参数，将使用默认值%f',parameters.decay));
    end
    
    %% 初始化
    T = parameters.max_it;
    W = parameters.window;
    beda1 = parameters.beda1;
    beda2 = parameters.beda2;
    r0 = parameters.learn_rate;
    D = parameters.decay;
    epsilon = parameters.epsilon;
    m = 0; % 初始化第一个递增向量
    v = 0; % 初始化第二个递增向量
    x1 = x0;  % 起始点
    y1 = F.object(x1,0); % 计算目标函数值
    
    %% 开始迭代
    z = y1; k = inf; 
    for it = 1:T
        r  = r0 * exp(-D*it/T); 
        g1 = F.gradient(x1,it); % 计算梯度
        m  = beda1 * m + (1 - beda1) * g1;    % 更新第1个增量向量
        v  = beda2 * v + (1 - beda2) * g1.^2; % 更新第2个增量向量
        mb  = m / (1 - beda1^it); % 对第1个增量向量进行修正
        vb  = v / (1 - beda2^it); % 对第2个增量向量进行修正
        x2 = x1 - r * mb ./ (sqrt(vb) + epsilon);
        y2 = F.object(x2,it); % 计算目标函数值
        z = (1-1/W)*z + (1/W)*y2;
        disp(sprintf('迭代次数:%d 学习速度:%f 目标均值:%f 目标函数:%f',it,r,z,y2));
        x1 = x2; y1 = y2;
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

