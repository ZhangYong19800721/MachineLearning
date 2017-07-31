function [x1,z1] = LevenbergMarquardt(F,J,x0,epsilon,max_it)
%LevenbergMarquardt 一种求解最小二乘问题的数值算法
%   是一种改进的高斯牛顿法，它的搜索方向介于高斯牛顿方向和最速下降方向之间

    alfa = 0.01; beda = 10; 
    D = length(x0); % 数据的维度
    x1 = x0; z1 = [];
    
    ob_window_size = 100;
    ob = learn.Observer('残差',1,ob_window_size,'xxx');
    
    for it = 1:max_it
        j1 = J.jacobi(x1);
        f1 = F.myfunc(x1);
        g1 = j1' * f1;
        ng = norm(g1);
        if ng < epsilon
            break;
        end
        delta = -1 * (j1'*j1 + alfa * eye(D)) \ g1;
        x2 = x1 + delta';
        f2 = F.myfunc(x2);
        z1 = norm(f1,2).^2;
        z2 = norm(f2,2).^2;
        
        description = strcat('迭代次数:',num2str(it));
        description = strcat(description,strcat(' 梯度模:',num2str(ng)));
        ob = ob.showit(z1,description);
        
        if z1 > z2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end

