function [ny,nx] = gold(F,a,b,parameters)
%gold 使用黄金分割法进行线搜索
%  输入：
%       F 单变量函数，F.object(x)计算目标函数在x处的值
%       a 搜索区间的左端点
%       b 搜索区间的右端点
%       parameters.epsilon 停止条件，默认值1e-6
%  输出：
%       ny 最小点的函数值
%       nx 最小点的变量值

    %% 参数设置
    if nargin <= 3 % 没有给出参数的情况
        parameters = [];
        disp('调用gold函数时没有给出参数集，将使用默认参数');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon的情况
        parameters.epsilon = 1e-6; 
        disp(sprintf('调用gold函数时参数集中没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end
  
    %% 使用黄金分割法进行一维搜索
    g = (sqrt(5)-1)/2;
    ax = a + (1 - g)*(b - a); Fax = F.object(ax);
    bx = a + g * (b - a);     Fbx = F.object(bx);
    
    while b - a > parameters.epsilon
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

