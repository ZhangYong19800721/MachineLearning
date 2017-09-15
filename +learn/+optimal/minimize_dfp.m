function [x,y] = minimize_dfp(F,x0,parameters)
%MINIMIZE_DFP DFP拟牛顿法求解无约束最优化问题
%   参考 马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社） 

    %% 参数设置
    if nargin <= 2 % 没有给出参数
        parameters = [];
        disp('调用minimize_dfp函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon
        parameters.epsilon = 1e-5; 
        disp(sprintf('调用minimize_dfp函数时参数集中没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % 给出参数但是没有给出max_it
        parameters.max_it = 1e6;
        disp(sprintf('调用minimize_dfp函数时参数集中没有max_it参数，将使用默认值%f',parameters.max_it));
    end
    
    if ~isfield(parameters,'armijo') % 给出参数但是没有给出armijo
        parameters.armijo = [];
    end

    %%
    x1 = x0;
    D = length(x0); H = eye(D); % D参数维度，B是Hessen矩阵的近似逆矩阵，起始时为单位阵
    G1 = F.gradient(x1); y1 = F.object(x1); % 计算x1处的梯度和函数值
    for it = 0:parameters.max_it
        ng1 = norm(G1); % 计算梯度模
        disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f',y1,it,ng1));
        if ng1 < parameters.epsilon, break; end   %当梯度模足够小时终止迭代
        d = -H * G1;  % 计算搜索方向
        [y2,x2] = learn.optimal.armijo(F,x1,G1,d,parameters.armijo);
        G2 = F.gradient(x2);  %计算x2处的梯度
        s = x2 - x1;
        z = G2 - G1;
        if z' * s > 0
            H = H - (H*(z*z')*H)/(z'*H*z)+(s*s')/(s'*z);
        end
        x1 = x2; G1 = G2; y1 = y2;
    end
    x = x1;
    y = y1;
end

