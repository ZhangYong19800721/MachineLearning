classdef GentleAdaBoost
    %GENTLEADABOOST Gentle AdaBoost算法实现
    %  参考论文"Additive Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了AdaBoost、RealAdaBoost、LogitBoost、GentleAdaBoost算法。
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods(Access = private)
         %% 选择最优弱分类器
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump 在给定维度的情况下选择最优的stump参数
            % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * (x(k) > t) + b
            % 输入：
            %   points 数据点（只包含第k维的数据，只有一行）
            %   labels 标签,+1或-1
            %   weight 权值,和为1的非负向量
            % 输出：
            %   t 门限
            %   a a+b等于在门限值右侧的(加权激活概率-加权非激活概率)
            %   b b  等于在门限值左侧的(加权激活概率-加权非激活概率)
            %   err 误差值 err = sum(w * |z - (a*(x>th) + b)|^2)
            
            %% 初始化
            [K,N] = size(points); % K数据的维度，N数据点数
            
            %% 计算所有可能的门限值T 
            % 将标签和权值跟随点值进行排序
            [T, sort_idx] = sort(points); 
            T = [T(1)-eps,T+eps]; % 得到所有可能的门限值
            
            %% 对所有可能的门限值计算a和b的值
            l = labels(sort_idx); w = weight(sort_idx); % 标签和权值跟随排序
            Szw = [0 cumsum(l.*w)]; Ezw = Szw(end); Sw  = [1e-100 cumsum(w)]; % 计算累计和
            B = Szw ./ Sw; % 所有可能的b值
            A = (Ezw - Szw) ./ max((1-Sw),1e-100) - B; % 所有可能的A值
            
            %% 计算误差
            % 误差的计算方式为
            %   error = sum(w.*(z-(a(i)*(x>th(i))+b(i))).^2);
            % 实际计算中使用下面效率更高的计算方式
            err = sum(w.*l.^2) - 2*A.*(Ezw-Szw) - 2*B*Ezw + (A.^2 +2*A.*B) .* (1-Sw) + B.^2;
            
            % 输出参数并找最优的门限
            [err, best.i] = min(err);
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
            t = zeros(1,K); a = zeros(1,K); b = zeros(1,K); err = zeros(1,K);
            
            %% 对每一个维度，计算最优的stump参数
            for k = 1:K
                [t(k),a(k),b(k),err(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% 设定stump的参数
            wc = learn.Stump();
            [~, wc.k] = min(err);
            wc.t = t(wc.k); wc.a = a(wc.k); wc.b = b(wc.k);
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
            F = sum(f,1);
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
                
                %% 计算弱分类器的输出，并更新强分类器的函数值
                fm = wc.compute(points); % 计算弱分类器的输出
                Fx = Fx + fm; % 更新强分类器
                
                %% 更新并归一化权值
                weight = weight .* exp(-labels.*fm); % 更新权值
                weight = weight ./ sum(weight); % 归一化权值
                
                %% 计算错误率
                y = obj.predict(points);
                disp(sum(xor(y,labels>0)) / N);
            end
        end
    end
    
    %% 单元测试
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(2)
            
            boost = learn.GentleAdaBoost();
            
            N = 1e4;
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

