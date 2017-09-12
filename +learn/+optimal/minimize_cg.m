function [x1,y1] = minimize_cg(F,x0,parameters)
%minimize_cg 共轭梯度法
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数在x处的值，调用F.gradient(x)计算目标函数在x处的梯度
%   x0 迭代的起始位置
%   parameters.epsilon 当梯度模小于epsilon时停止迭代
%   parameters.max_it 最大迭代次数
%   parameters.reset 重置条件
%   parameters.option 线搜索方法 
%       'gold' 黄金分割法（精确搜索）
%       'armijo' Armijo准则（非精确搜索）
%   当采用黄金分割法进行搜索时：
%       parameters.gold.epsilon 线性搜索的停止条件
%   当采用Armijo进行搜索时：
%       parameters.armijo.beda 值必须在(0,  1)之间
%       parameters.armijo.alfa 值必须在(0,0.5)之间
%       parameters.armijo.maxs 最大搜索步数，正整数

    %ob = learn.tools.Observer('函数值',1,100);
    %% 计算起始位置的函数值、梯度、梯度模
    x1 = x0; y1 = F.object(x1); g1 = F.gradient(x1); ng1 = norm(g1); % 起始点为x0,计算函数值、梯度、梯度模 
    if ng1 < parameters.epsilon, return; end % 如果梯度足够小，直接返回
    
    %% 迭代寻优
    d1 = -g1; % 初始搜索方向为负梯度方向
    for it = 1:parameters.max_it
        if ng1 < parameters.epsilon, return; end % 如果梯度足够小，直接返回
        
        % 沿d1方向线搜索
        if strcmp(parameters.option,'gold') 
            [a,b] = learn.optimal.AR(learn.optimal.LINE(F,x1,d1),0,1);
            parameters.gold.a = a; parameters.gold.b = b;
            [~,y2,x2] = learn.optimal.gold(F,x1,d1,parameters);
        elseif strcmp(parameters.option,'armijo')
            [~,y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters);
        end
        
        c1 = mod(it,parameters.reset) == 0; % 到达重置点
        c2 = y1 < y2; %表明d1方向不是一个下降方向
        if c1 || c2
            d1 = -g1; % 设定搜索方向为负梯度方向
            if strcmp(parameters.option,'gold')
                [a,b] = learn.optimal.AR(learn.optimal.LINE(F,x1,d1),0,1);
                parameters.gold.a = a; parameters.gold.b = b;
                [~,y2,x2] = learn.optimal.gold(F,x1,d1,parameters);
            elseif strcmp(parameters.option,'armijo')
                [~,y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters);
            end
            g2 = F.gradient(x2); d2 = -g2; ng2 = norm(g2); % 迭代到新的位置x2，并计算函数值、梯度、搜索方向、梯度模
            x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
            disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f ',y1,it,ng1));
            continue;
        end
 
        g2 = F.gradient(x2); ng2 = norm(g2); % 计算x2处的梯度和梯度模
        beda = g2'*(g2-g1)/(g1'*g1); d2 = -g2 + beda * d1; % 计算x2处的搜索方向d2
        x1 = x2; d1 = d2; g1 = g2; y1 = y2; ng1 = ng2;
        disp(sprintf('目标函数:%f 迭代次数:%d 梯度模:%f ',y1,it,ng1));
    end
end

