classdef BoostSSC
    %BoostSSC Boost Similarity Sensitive Coding算法实现
    %  参考论文"Small Codes and Large Image Databases for Recognition" Section
    %  2.1
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods(Access = private)
         %% 选择最优弱分类器
        function [t,a,b,err] = select_stump(obj,points,labels,weight)
            %select_stump 在给定维度的情况下选择最优的stump参数
            % 弱分类器器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % 输入：
            %   points 数据点（只包含第k维的数据，只有一行）
            %   labels 标签,共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每一列表示一个样本对)
            %   weight 权值,和为1的非负向量
            % 输出：
            %   t 门限
            %   a 在门限值右侧的 加权激活概率 - 加权非激活概率
            %   b 在门限值左侧的 加权激活概率 - 加权非激活概率
            %   err 误差值 err = sum(w * |z - (a*[(x1(k)>t)==(x2(k)>t)] + b)|^2)
            
            %% 初始化
            [K,N] = size(points); % K数据的维度，N数据点数
            t = zeros(1,K); a = zeros(1,K); b = zeros(1,K); err = zeros(1,K);
            
            %% 将标签和权值跟随点值进行排序
            [x, sort_idx] = sort(points); % 对点值进行排序，将被作为门限使用
            l = labels(sort_idx); w = weight(sort_idx); % 标签和权值跟随排序
            
            %% 计算累计和
            Szw = cumsum(l.*w); Ezw = Szw(end); Sw  = cumsum(w);
            
            %% 对所有可能的门限值计算a和b的值
            b = Szw ./ Sw; % 所有可能的b值
            zz = Sw == 1;
            Sw(zz) = 0;
            a = (Ezw - Szw) ./ (1-Sw) - b; % 所有可能的a值
            Sw(zz) = 1;
            
            %% 计算误差
            % 误差的计算方式为
            %   error = sum(w.*(z-(a(i)*(x>th(i))+b(i))).^2);
            % 实际计算中使用下面效率更高的计算方式
            err = sum(w.*l.^2) - 2*a.*(Ezw-Szw) - 2*b*Ezw + (a.^2 +2*a.*b) .* (1-Sw) + b.^2;
            
            % 输出参数并找最优的门限
            [err, n] = min(err);
            
            if n == N
                t = x(n);
            else
                t = (x(n) + x(n+1))/2;
            end
            a = a(n);
            b = b(n);
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
            [K,N] = size(points); % K 数据的维度，N数据点数
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
            [~,N] = size(points1); % N数据点数
            f = zeros(M,N); % 存储弱分类器的分类结果
            
            %% 计算所有弱分类器的输出
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points1,points2); 
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
                fm = wc.compute(points,pairidx); % 计算弱分类器的输出
                Fx = Fx + fm; % 更新强分类器
                
                %% 更新并归一化权值
                weight = weight .* exp(-labels.*fm); % 更新权值
                weight = weight ./ sum(weight); % 归一化权值
                
                %% 计算错误率
                y = obj.predict(points,pairs);
                err(m) = sum(xor(y,labels>0)) / N
            end
        end
    end
    
    %% 单元测试
    methods(Static)
        function ssc = unit_test()
            clear all;
            close all;
            rng(2)
            
            ssc = learn.BoostSSC();
            
            N = 1e4;
            [points,labels] = learn.GenerateData.type4(N);
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'.'); 
            
            M = 64;
            ssc = ssc.train(points,labels,M);
        end
    end
end

