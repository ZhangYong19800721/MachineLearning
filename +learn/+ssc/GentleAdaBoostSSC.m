classdef GentleAdaBoostSSC
    %GentleAdaBoostSSC Gentle AdaBoost Similarity Sensitive Coding算法实现
    %  参考论文"Small Codes and Large Image Databases for Recognition" Section
    %  2.1
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods(Access = private)
         %% 选择最优弱分类器
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump 在给定维度的情况下选择最优的stump参数
            % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % 输入：
            %   points 数据点（只包含第k维的数据，只有一行）
            %   labels 标签,共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每一列表示一个样本对)
            %   weight 权值,和为1的非负向量，与labels的列数相同
            % 输出：
            %   t 门限
            %   a a+b表示被弱分类器判为正例的所有数据点的（加权激活概率 - 加权非激活概率）
            %   b   b表示被弱分类器判为反例的所有数据点的（加权激活概率 - 加权非激活概率）
            %   err 误差值 err = sum(w * |z - (a*[(x1(k)>t)==(x2(k)>t)] + b)|^2)
            
            %% 初始化
            I = labels(1,:);      % 数据序列1
            J = labels(2,:);      % 数据序列2
            L = labels(3,:);      % 标签
            
            %% 计算所有可能的门限值T
            T = unique(points); % 去除重复值并排序
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2,T(end)+eps]; % 得到所有可能的门限值
            
            %% 对所有可能的门限值计算a和b的值
            A = zeros(size(T)); B = zeros(size(T));
            for m = 1:length(T)
                t = T(m); pos = points > t; i = pos(I)==pos(J);
                B(m) = sum(L(~i) .* weight(~i)) / max(1e-100,sum(weight(~i)));
                A(m) = sum(L( i) .* weight( i)) / max(1e-100,sum(weight( i))) - B(m);
            end
            
            %% 计算误差
            % 误差的计算方式为
            %   error = sum(w.*(z-(a(i)*[(x1>t(i))==(x2>t(i))]+b(i))).^2);
            E = zeros(size(T)); x1 = points(I); x2 = points(J); 
            for m = 1:length(T)
                E(m) = sum(weight .* (L - (A(m)*((x1>T(m))==(x2>T(m)))+B(m))).^2);
            end
            
            %% 输出参数并找最优的门限
            [err, best.i] = min(E);
            t = T(best.i);
            a = A(best.i);
            b = B(best.i);
        end
        
        %% 选择最优弱分类器
        function wc = select_wc(obj,points,labels,weight)
            %select_wc 选择最优的stump弱分类器
            % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % 输入：
            %   points 数据点
            %   labels 标签,共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每一列表示一个样本对)
            %   weight 权值,和为1的非负向量
            % 输出：
            %   wc 最优的弱分类器
            
            %% 初始化
            [K,N] = size(points); % K数据的维度，N数据点数
            [~,Q] = size(labels); % 数据对的数目
            t = zeros(1,K); a = zeros(1,K); b = zeros(1,K); err = zeros(1,K);
            
            %% 对每一个维度，计算最优的stump参数
            for k = 1:K
                [t(k),a(k),b(k),err(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            
            %% 设定stump的参数
            wc = learn.StumpSSC();
            [~, wc.k] = min(err);
            wc.t = t(wc.k); wc.a = a(wc.k); wc.b = b(wc.k);
        end
    end
    
    methods(Access = public)
        %% 哈希编码
        function c = hash(obj,points)
            %hash 计算数据点的哈希码
            % 该hash码是保留了相似性的hash码 
            
            %% 初始化
            M = length(obj.weak); % M弱分类器的个数
            [~,N] = size(points); % N数据点数
            c = false(M,N);       % 存放hash码的变量，每一列对应一个数据点
            
            %% 计算数据点对应的M个比特
            for m = 1:M
                k = obj.weak{m}.k; t = obj.weak{m}.t;
                c(m,:) = points(k,:) > t;
            end
        end
        
        %% 判决
        function [y,F] = predict(obj,points,pairs)
            %PREDICT 判断两个数据点是否相似，相似为正1，不相似为反0
            % 
            
            %% 初始化
            M = length(obj.weak); % M弱分类器的个数
            [~,Q] = size(pairs);  % Q数据对数
            f = zeros(M,Q); % 存储弱分类器的分类结果
            I = pairs(1,:);
            J = pairs(2,:);
            
            %% 计算所有弱分类器的输出
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points,pairs);
            end
            
            %% 累加弱分类器的输出并判决
            F = sum(f,1);
            y = F > 0;
        end
        
        %% 训练
        function obj = train(obj,points,labels,M)
            % train 训练BoostSSC模型
            % 输入：
            % points 数据点
            % labels 标签，共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每1列表示一个样本对)
            % M 弱分类器的个数
            % 输出：
            % obj 经过训练的BoostSCC对象
            
            %% 初始化
            [~,N] = size(points); % 得到数据点数目
            [~,Q] = size(labels); % 得到数据对数目
            I = labels(1,:);
            J = labels(2,:);
            L = labels(3,:);
            P = [I;J];
            Fx = zeros(1,Q); % 用来记录函数Fx的值
            weight = ones(1,Q) / Q; % 初始化权值
            
            %% 迭代
            for m = 1:M
                %% 选择最优的弱分类器
                % 弱分类器器fm是一个stump函数，由4个参数确定：(a,b,k,t)
                % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
                wc = obj.select_wc(points,labels,weight);
                
                %% 将最优弱分类器加入到强分类器中
                obj.weak{1+length(obj.weak)} = wc;
                
                %% 计算弱分类器的输出，并更新强分类器的函数值
                fm = wc.compute(points,P); % 计算弱分类器的输出
                Fx = Fx + fm; % 更新强分类器
                
                %% 更新并归一化权值
                weight = weight .* exp(-L.*fm); % 更新权值
                weight = weight ./ sum(weight); % 归一化权值
                
                %% 计算错误率
                y = obj.predict(points,P); l = L>0;
                error = sum(y~=l) / Q;
                disp(num2str(error));
                
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
        function ssc = unit_test()
            clear all;
            close all;
            rng(2)
            
            ssc = learn.GentleAdaBoostSSC();
            
            N = 500;
            [points,labels] = learn.GenerateData.type6(N);
            plot(points(1,:),points(2,:),'.');hold on;
            
            M = 10;
            ssc = ssc.train(points,labels,M);
        end
    end
end

