function [x1,z1] = minimize_lm(F,x0,parameters)
%minimize_lm LevenbergMarquardt算法，一种求解最小二乘问题的数值算法
%   是一种改进的高斯牛顿法，它的搜索方向介于高斯牛顿方向和最速下降方向之间
%   输入：
%       F 调用[H,G]=F.hessian(x)计算目标函数的近似Hessian矩阵（H=J'*J,J为Jacobi矩阵）和梯度G，
%         将H和G一起计算出来是因为Jacobi矩阵的计算量大且容量大无法存储
%         调用F.object(x) 计算目标函数的值
%       x0 迭代的起始位置
%       parameters 参数集
%   输出：
%       x 最优的参数解
%       y 最小的函数值

    %% 参数设置
    if nargin <= 2 % 没有给出参数
        parameters = [];
        disp('调用minimize_lm函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('调用minimize_lm函数时参数集中没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % 给出参数但是没有给出max_it
        parameters.max_it = 1e6;
        disp(sprintf('调用minimize_lm函数时参数集中没有max_it参数，将使用默认值%d',parameters.max_it));
    end

    %%
    alfa = 0.01; beda = 10; 
    D = length(x0); % 数据的维度
    x1 = x0;
    
    for it = 1:parameters.max_it
        [H1,G1] = F.hessen(x1);
        F1 = F.object(x1);
        ng1 = norm(G1);
        if ng1 < parameters.epsilon
            break;
        end
        delta = -(H1 + alfa * eye(D)) \ (G1/2);
        x2 = x1 + delta;
        F2 = F.object(x2);
        
        disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f',F1,it,ng1));
        
        if F1 > F2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end

