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
%       'parabola' 抛物线搜索法（精确搜索）
%       'armijo' Armijo准则（非精确搜索）
%   当采用黄金分割法进行搜索时：
%       parameters.gold.epsilon 线性搜索的停止条件
%   当采用Armijo进行搜索时：
%       parameters.armijo.beda 值必须在(0,  1)之间，典型值0.5
%       parameters.armijo.alfa 值必须在(0,0.5)之间，典型值0.2
%       parameters.armijo.maxs 最大搜索步数，正整数，典型值30
%   当采用抛物线法进行搜索时：
%       parameters.parabola.epsilon 线性搜索的停止条件

    %% 参数设置
    if nargin <= 2 % 没有给出参数
        parameters = [];
        disp('调用minimize_cg函数时没有给出参数集，将使用默认参数集');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('调用minimize_cg函数时参数集中没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % 给出参数但是没有给出max_it
        parameters.max_it = 1e6;
        disp(sprintf('调用minimize_cg函数时参数集中没有max_it参数，将使用默认值%d',parameters.max_it));
    end
    
    if ~isfield(parameters,'reset') % 给出参数但是没有给出reset
        parameters.reset = 500;
        disp(sprintf('调用minimize_cg函数时参数集中没有reset参数，将使用默认值%f',parameters.reset));
    end
    
    if ~isfield(parameters,'option') % 给出参数但是没有给出option
        parameters.option = 'gold';
        disp(sprintf('调用minimize_cg函数时参数集中没有option参数，将使用默认值%s',parameters.option));
    end
    
    if ~isfield(parameters,'gold') % 给出参数但是没有给出gold
        parameters.gold = [];
    end
    
    if ~isfield(parameters,'parabola') % 给出参数但是没有给出parabola
        parameters.parabola = [];
    end
    
    if ~isfield(parameters,'armijo') % 给出参数但是没有给出armijo
        parameters.armijo = [];
    end

    %% 计算起始位置的函数值、梯度、梯度模
    x1 = x0; y1 = F.object(x1); g1 = F.gradient(x1); ng1 = norm(g1); % 起始点为x0,计算函数值、梯度、梯度模 
    if ng1 < parameters.epsilon, return; end % 如果梯度足够小，直接返回
    
    %% 迭代寻优
    d1 = -g1; % 初始搜索方向为负梯度方向
    for it = 1:parameters.max_it
        if ng1 < parameters.epsilon, return; end % 如果梯度足够小，直接返回
        
        % 沿d1方向线搜索
        if strcmp(parameters.option,'gold') % 黄金分割法进行一维精确线搜索
            Fs = learn.optimal.SINGLEX(F,x1,d1); % 包装为单变量函数
            [a,b] = learn.optimal.ARR(Fs,0,1,parameters.gold); % 确定搜索区间
            [y2,lamda] = learn.optimal.gold(Fs,a,b,parameters.gold); x2 = x1 + lamda * d1;
        elseif strcmp(parameters.option,'parabola') % 使用抛物线法进行一维搜索
            Fs = learn.optimal.SINGLEX(F,x1,d1); % 包装为单变量函数
            [a,b] = learn.optimal.ARR(Fs,0,1,parameters.parabola); % 确定搜索区间
            [y2,lamda] = learn.optimal.parabola(Fs,a,b,parameters.parabola); x2 = x1 + lamda * d1;
        elseif strcmp(parameters.option,'armijo') % armijo准则进行一维非精确搜索
            [y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters.armijo);
        end
        
        c1 = mod(it,parameters.reset) == 0; % 到达重置点
        c2 = y1 <= y2; %表明d1方向不是一个下降方向
        if c1 || c2
            d1 = -g1; % 设定搜索方向为负梯度方向
            if strcmp(parameters.option,'gold') % 黄金分割法进行一维精确线搜索
                Fs = learn.optimal.SINGLEX(F,x1,d1); % 包装为单变量函数
                [a,b] = learn.optimal.ARR(Fs,0,1,parameters.gold); % 确定搜索区间
                [y2,lamda] = learn.optimal.gold(Fs,a,b,parameters.gold); x2 = x1 + lamda * d1;
            elseif strcmp(parameters.option,'parabola') % 使用抛物线法进行一维搜索
                Fs = learn.optimal.SINGLEX(F,x1,d1); % 包装为单变量函数
                [a,b] = learn.optimal.ARR(Fs,0,1,parameters.parabola); % 确定搜索区间
                [y2,lamda] = learn.optimal.parabola(Fs,a,b,parameters.parabola); x2 = x1 + lamda * d1;
            elseif strcmp(parameters.option,'armijo') % armijo准则进行一维非精确搜索
                [y2,x2] = learn.optimal.armijo(F,x1,g1,d1,parameters.armijo);
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

