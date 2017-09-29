function [x,y] = minimize_g(F,x0,parameters)
%minimize_g 梯度法
%   输入：
%       F 调用F.gradient(x)计算目标函数的梯度，调用F.object(x)计算目标函数的值
%       x0 迭代的起始位置
%       parameters 参数集
%   输出：
%       x 最优的参数解
%       y 最小的函数值

    %% 参数设置
    if nargin <= 2 % 没有给出参数
        parameters = [];
        disp('调用minimize_g函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('没有epsilon参数，将使用默认值%f',parameters.epsilon));
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
        parameters.learn_rate = 0.1;
        disp(sprintf('没有learn_rate参数，将使用默认值%f',parameters.learn_rate));
    end
    
    %% 初始化
    m = parameters.momentum;
    r = parameters.learn_rate;
    inc_x = zeros(size(x0)); % 参数的递增量
    x1 = x0;  
    y1 = F.object(x1); % 计算目标函数值
    
    %% 开始迭代
    for it = 1:parameters.max_it
        g1 = F.gradient(x1); % 计算梯度
        ng1 = norm(g1); % 计算梯度模
        disp(sprintf('迭代次数:%d 学习速度:%f 目标函数:%f 梯度模:%f ',it,parameters.learn_rate,y1,ng1));
        if ng1 < parameters.epsilon
            break; % 如果梯度足够小就结束迭代
        end
        inc_x = m * inc_x - (1 - m) * r * g1; % 向负梯度方向迭代，并使用动量参数
        x1 = x1 + inc_x; % 更新参数值
        y1 = F.object(x1); % 计算目标函数值
    end
    
    %% 返回
    x = x1;
    y = y1;
end

