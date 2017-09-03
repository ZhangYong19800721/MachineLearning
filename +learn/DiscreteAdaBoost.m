classdef DiscreteAdaBoost
    %DiscreteAdaBoost DiscreteAdaBoost算法实现
    %  参考论文"Additive Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了AdaBoost、RealAdaBoost、LogitBoost、GentleAdaBoost算法。
    
    properties
        weak; % 包含若干个弱分类器的cell数组
        alfa; % 每个弱分类器的投票权值
    end
    
    methods(Access = private)
         %% 选择最优弱分类器
        function [t,a,b,z] = select_stump(obj,points,labels,weight)
            %select_t 在给定维度的情况下选择最优的门限值
            % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % 对于DiscreteAdaBoost(a=+2，b=-1)或(a=-2,b=+1)这样stump函数返回的值只能是+1或-1
            % fm = a * (x(k) > t) + b
            % 输入：
            %   points 数据点（只包含第k维的数据，只有一行）
            %   labels 标签,+1或-1
            %   weight 权值,和为1的非负向量
            % 输出：
            %   t 门限
            %   a+b 
            %   b
            %   z 弱分类器优化的目标函数z=sum(weight.*labels.*fm)
            
            %% 初始化
            [K,N] = size(points); % K数据的维度，N数据点数
            
            %% 计算所有可能的门限值T 
            T = unique(points); % 去除重复值并排序
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2, T(end)+eps]; % 得到所有可能的门限值
            
            %% 对所有可能的门限值计算a,b,z
            A = zeros(size(T)); B = zeros(size(T)); Z = zeros(size(T));
            for m = 1:length(T)
                t = T(m); 
                p =  2 * (points > t) - 1; 
                q = -2 * (points > t) + 1;
                Z1 = sum(weight.*labels.*p);
                Z2 = sum(weight.*labels.*q);
                if Z1 > Z2
                    Z(m) = Z1; A(m) =  2; B(m) = -1;
                else
                    Z(m) = Z2; A(m) = -2; B(m) =  1;
                end
            end
            
            %% 输出参数并找最优的门限
            [z, best.i] = max(Z);
            t = T(best.i);
            a = A(best.i);
            b = B(best.i);
        end
        
        %% 选择最优弱分类器
        function wc = select_wc(obj,points,labels,weight)
            %select_wc 选择最优的stump弱分类器
            % 弱分类器器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * (x(k) > t) + b
            % 输入：
            %   points 数据点
            %   labels 标签,+1或-1
            %   weight 权值
            % 输出：
            %   wc 最优的弱分类器
            
            %% 初始化
            [K,N] = size(points); % K 数据的维度，N数据点数
            T = zeros(1,K); A = zeros(1,K); B = zeros(1,K); Z = zeros(1,K);
            
            %% 对每一个维度，计算最优的stump参数
            for k = 1:K
                [T(k),A(k),B(k),Z(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% 设定stump的参数
            wc = learn.Stump();
            [~, wc.k] = max(Z);
            wc.t = T(wc.k); wc.a = A(wc.k); wc.b = B(wc.k);
        end
    end
    
    methods(Access = public)
        %% 判决
        function [y,F] = predict(obj,points)
            %PREDICT 对数据点进行分类，正例为1，反例为0
            % 使用经过训练的模型判断数据点的分类
    
            %% 初始化
            M = length(obj.weak); % M弱分类器的个数
            [~,N] = size(points); % N数据点数
            f = zeros(M,N); % 存储弱分类器的分类结果
            
            %% 计算所有弱分类器的输出
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points); 
            end
            
            %% 累加弱分类器的输出并判决
            F = obj.alfa * f;
            y = F > 0;
        end
        
        %% 训练
        function obj = train(obj,points,labels,M)
            % train 训练GentleAdaBoost模型
            % 输入：
            % points 数据点
            % labels 标签，+1或-1
            % M 弱分类器的个数
            % 输出：
            % obj 经过训练的boost对象
            
            %% 初始化
            [~,N] = size(points); % 得到数据点数目
            Fx = zeros(1,N); % 用来记录函数Fx的值
            weight = ones(1,N) / N; % 初始化权值
            
            %% 迭代
            for m = 1:M
                %% 选择最优的弱分类器
                % 弱分类器器fm是一个stump函数，由4个参数确定：(a,b,k,t)
                % fm = a * (x(k) > t) + b
                wc = obj.select_wc(points,labels,weight);
                
                %% 将最优弱分类器加入到强分类器中
                obj.weak{1+length(obj.weak)} = wc;
                
                %% 计算弱分类器的输出，计算加权值并更新强分类器的函数值
                fm = wc.compute(points); % 计算弱分类器的输出
                err = weight * xor((fm > 0),(labels > 0))';
                err = max(err,1e-100);
                c = 0.5 * log((1 - err)/err);
                obj.alfa = [obj.alfa c];
                Fx = Fx + c.*fm; % 更新强分类器
                
                %% 更新并归一化权值
                weight = weight .* exp(-c.*labels.*fm); % 更新权值
                weight = weight ./ sum(weight); % 归一化权值
                
                %% 计算错误率
                disp(sum(xor(Fx>0,labels>0)) / N);
                
                %% 画图
                if wc.k == 1
                    x0 = 1; y0 = 0; z0 = -wc.t;
                else
                    x0 = 0; y0 = 1; z0 = -wc.t;
                end
                f = @(x,y) x0*x+y0*y+z0;
                ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
                drawnow;
            end
        end
    end
    
    %% 单元测试
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(2)
            
            boost = learn.DiscreteAdaBoost();
            
            N = 1e3;
            [points,labels] = learn.GenerateData.type4(N);
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'.'); 
            
            M = 300;
            boost = boost.train(points,labels,M);
        end
    end
end

