function [ny,nx] = gold(F,a,b,parameters)
%gold 使用黄金分割法进行线搜索
%  输入：
%       F 单变量函数，F.object(x)计算目标函数在x处的值
%       a 搜索区间的左端点
%       b 搜索区间的右端点
%       parameters.gold.epsilon 停止条件，典型值1e-6
%  输出：
%       ny 最小点的函数值
%       nx 最小点的变量值

    if nargin <= 3
        parameters.gold.epsilon = 1e-6;
    end
    
    g = (sqrt(5)-1)/2;
    ax = a + (1 - g)*(b - a); Fax = F.object(ax);
    bx = a + g * (b - a);     Fbx = F.object(bx);
    
    while b - a > parameters.gold.epsilon
        if Fax > Fbx
            a = ax;
            ax = bx; Fax = Fbx;
            bx = a + g * (b - a);
            Fbx = F.object(bx);
        else
            b = bx;
            bx = ax; Fbx = Fax;
            ax = a + (1 - g)*(b - a);
            Fax = F.object(ax);
        end
    end
    
    if Fax > Fbx
        nx = bx; ny = Fbx;
    else
        nx = ax; ny = Fax;
    end
end

