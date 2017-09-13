function [x1,z1] = minimize_lm(F,x0,parameters)
%minimize_lm LevenbergMarquardt算法，一种求解最小二乘问题的数值算法
%   是一种改进的高斯牛顿法，它的搜索方向介于高斯牛顿方向和最速下降方向之间
%   输入：
%       F 调用F.jacobi(x)计算目标函数的jacobi矩阵，调用F.vector(x)计算目标函数的各个分项值f = [f1,f2,f3,...,fn]
%       目标函数为sum(f.^2)
%       x0 迭代的起始位置
%       parameters.epsilon 当梯度的范数小于epsilon时迭代结束
%       parameters.max_it 最大迭代次数
%   输出：
%       x 最优的参数解
%       y 最小的函数值

    alfa = 0.01; beda = 10; 
    D = length(x0); % 数据的维度
    x1 = x0;
    
    for it = 1:parameters.max_it
        j1 = F.jacobi(x1);
        f1 = F.vector(x1);
        g1 = j1' * f1;
        ng1 = norm(g1);
        if ng1 < parameters.epsilon
            break;
        end
        delta = -1 * (j1'*j1 + alfa * eye(D)) \ g1;
        x2 = x1 + delta;
        f2 = F.vector(x2);
        z1 = norm(f1,2).^2;
        z2 = norm(f2,2).^2;
        
        disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f',sum(f1.^2),it,ng1));
        
        if z1 > z2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end

