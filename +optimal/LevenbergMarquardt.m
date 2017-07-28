function [x1,z1] = LevenbergMarquardt(f,J,x0,epsilon)
%LevenbergMarquardt 一种求解最小二乘问题的数值算法
%   是一种改进的高斯牛顿法，它的搜索方向介于高斯牛顿方向和最速下降方向之间
    alfa = 0.01; beda = 10; D = length(x0);
    
    x1 = x0; 
    
    while true
        J1 = J(x1); 
        f1 = f(x1);
        G1 = J1' * f1;
        if norm(G1) < epsilon
            break;
        end
        delta = -1 * (J1'*J1 + alfa * eye(D)) \ G1;
        x2 = x1 + delta;
        f2 = f(x2);
        z1 = norm(f1,2).^2;
        z2 = norm(f2,2).^2;
        if z1 > z2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end

