function [x,y] = minimize_g(F,x0,parameters)
%minimize_g 梯度法
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
    
    %% 初始化
    ob = learn.tools.Observer('目标函数值',1,100); % 初始化观察者
    inc_x = zeros(size(x0)); % 参数的递增量
    m = parameters.momentum;
    r = parameters.learn_rate;
    x1 = x0; 
    
    %% 开始迭代
    for it = 1:parameters.max_it
        g1 = F.gradient(x1); % 计算梯度
        y1 = F.object(x1); % 计算目标函数值
        ng = norm(g1); % 计算梯度模
        
        description = sprintf('学习速度：%f 迭代次数: %d 梯度模：%f ',r,it,ng);
        ob = ob.showit(y1,description);
        
        if ng < parameters.epsilon
            break; % 如果梯度足够小就结束迭代
        end
        
        inc_x = m * inc_x - (1 - m) * r * g1; % 向负梯度方向迭代，并使用动量参数
        x1 = x1 + inc_x; % 更新参数值
    end
    
    %% 返回
    x = x1;
    y = y1;
end

