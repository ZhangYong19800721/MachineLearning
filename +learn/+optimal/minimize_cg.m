function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg 共轭梯度法
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数在x处的值，调用F.gradient(x)计算目标函数在x处的梯度
%   x0 迭代的起始位置
%   parameters.epsilon 当梯度模小于epsilon时停止迭代
%   parameters.alfa alfa*epsilon是线性搜索的停止条件
%   parameters.max_it 最大迭代次数
%   parameters.reset 重置条件
%   parameters.dis 线性搜索的最大距离  

    x1 = x0; y1 = F.object(x0); % 起始点为x0,并计算初始的目标函数值 
    g1 = F.gradient(x1); % 计算x1处的梯度（此时x1=x0）
    d1 = -g1; % 初始搜索方向为负梯度方向
    ng = norm(g1); % 计算梯度模
    if ng < parameters.epsilon
        return;
    end
    
    alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.alfa*parameters.epsilon);
    
    for it = 1:parameters.max_it
        ng = norm(g1); % 计算梯度模
        if ng < parameters.epsilon
            break;
        end
        x2 = x1 + alfa * d1; 
        y2 = F.object(x2);
        if mod(it,parameters.reset) == 0 || y1 < y2
            d1 = -g1;
            alfa = learn.optimal.search(F,x1,d1,0,parameters.dis,parameters.alfa*parameters.epsilon);
            disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f ',y1,it,ng));
            continue;
        end
        
        disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f ',y1,it,ng));
        g2 = F.gradient(x2); % 计算x2处的梯度
        beda = g2'*(g2-g1)/(g1'*g1);
        d2 = -g2 + beda * d1;
        alfa = learn.optimal.search(F,x2,d2,0,parameters.dis,parameters.alfa*parameters.epsilon);
        x1 = x2; d1 = d2; g1 = g2; y1 = y2;
    end
end

