function [x1,y1] = ConjugateGradient(F,x0,epsilon1,epsilon2,max_it,reset,dis)
%CONJUGATEGRADIENT 共轭梯度法
%   
    ob = learn.Observer('目标函数值',1,100,'xxx');
    x1 = x0; y1 = F.object(x0); % 起始点为x0,并计算初始的目标函数值 
    g1 = F.gradient(x1); % 计算x1处的梯度（此时x1=x0）
    d1 = -g1; % 初始搜索方向为负梯度方向
    if norm(g1) < epsilon1
        return;
    end
    
    alfa = optimal.GoldenSection(F,x1,d1,0,dis,epsilon2);
    
    for it = 1:max_it
        if norm(g1) < epsilon1
            break;
        end
        x2 = x1 + alfa * d1; 
        y2 = F.object(x2);
        if mod(it,reset) == 0 || y1 < y2
            d1 = -g1;
            alfa = optimal.GoldenSection(F,x1,d1,0,dis,epsilon2);
            
            description = strcat(strcat(strcat('迭代次数:',num2str(it)),'/'),num2str(max_it));
            description = strcat(description,strcat(' 梯度模:',num2str(norm(g1))));
            ob = ob.showit(y1,description);
            continue;
        end
        
        description = strcat(strcat(strcat('迭代次数:',num2str(it)),'/'),num2str(max_it));
        description = strcat(description,strcat(' 梯度模:',num2str(norm(g1))));
        ob = ob.showit(y1,description);
        
        g2 = F.gradient(x2); % 计算x2处的梯度
        beda = g2'*(g2-g1)/(g1'*g1);
        d2 = -g2 + beda * d1;
        alfa = optimal.GoldenSection(F,x2,d2,0,dis,epsilon2);
        x1 = x2; d1 = d2; g1 = g2; y1 = y2;
    end
end

