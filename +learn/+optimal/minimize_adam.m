function [x,y] = minimize_adam(F,x0,parameters)
%minimize_adam ADAM随机梯度下降
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
    
    if ~isfield(parameters,'omiga') % 给出参数但是没有给出omiga
        parameters.omiga = 1e-6; 
        disp(sprintf('没有omiga参数，将使用默认值%f',parameters.omiga));
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
    
    %% 初始化
    m = 0; % 初始化第一个递增向量
    v = 0; % 初始化第二个递增向量
    x1 = x0;  % 起始点
    y1 = F.object(x1); % 计算目标函数值
    
    %% 开始迭代
    for it = 1:parameters.max_it
        g1 = F.gradient(x1,it); % 计算梯度
        m  = parameters.beda1 * m + (1 - parameters.beda1) * g1;    % 更新第1个增量向量
        v  = parameters.beda2 * v + (1 - parameters.beda2) * g1.^2; % 更新第2个增量向量
        mb  = m / (1 - parameters.beda1^it); % 对第1个增量向量进行修正
        vb  = v / (1 - parameters.beda2^it); % 对第2个增量向量进行修正
        x2 = x1 - parameters.learn_rate * mb ./ (sqrt(vb) + parameters.epsilon);
        y2 = F.object(x2); % 计算目标函数值
        disp(sprintf('迭代次数:%d 学习速度:%f 目标函数:%f 函数降幅:%f ',it,parameters.learn_rate,y2,abs(y1-y2)));
        if abs(y1 - y2) < parameters.omiga
            x1 = x2; y1 = y2;
            break; % 如果梯度足够小就结束迭代
        end
        x1 = x2; y1 = y2;
    end
    
    %% 返回
    x = x1;
    y = y1;
end

