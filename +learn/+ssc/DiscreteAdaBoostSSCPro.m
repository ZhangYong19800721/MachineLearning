classdef DiscreteAdaBoostSSCPro
    %DiscreteAdaBoostSSCPro DiscreteAdaBoostSSCPro算法实现
    %  参考论文"Learning Task-specific Similarity" 2005, Chapter 3, by Gregory
    %  Shakhnarovich
    
    properties
        weak; % 包含若干个弱分类器的cell数组
        alfa; % 每个弱分类器的投票权值
    end
    
    methods(Access = private)
         %% 选择最优弱分类器
        function [t,z] = select_stump(obj,points,labels,weight)
            %select_stump 在给定维度的情况下选择最优的stump参数
            % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
            % fm = a * [(x1(k) > t) == (x2(k) > t)] + b
            % 对于DiscreteAdaBoost(a=+2，b=-1)这样stump函数返回的值只能是+1或-1
            % 输入：
            %   points 数据点（只包含第k维的数据，只有一行）
            %   labels 标签,共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每一列表示一个样本对)
            %   weight 权值,和为1的非负向量，与labels的列数相同
            % 输出：
            %   t 门限
            %   z 弱分类器优化的目标函数z=sum(weight.*labels.*fm)
            
            %% 初始化
            [K,N] = size(points); % K数据的维度，N数据点数
            I = labels(1,:); J = labels(2,:); L = labels(3,:); P = [I;J];
            
            %% 计算所有可能的门限值T 
            T = unique(points); % 去除重复值并排序
            T = [T(1)-eps, (T(1:(end-1)) + T(2:end))/2, T(end)+eps]; % 得到所有可能的门限值
            
            %% 对所有可能的门限值计算a,b,z
            Z = zeros(size(T));
            for m = 1:length(T)
                t = T(m); pos = points > t;
                p =  2 * (pos(I) == pos(J)) - 1; 
                Z(m) = sum(weight.*L.*p);
            end
            
            %% 输出参数并找最优的门限
            [~, best.i] = max(abs(Z));
            z = Z(best.i); t = T(best.i);
        end
        
        %% 选择最优弱分类器
        function wc = select_wc(obj,points,labels,weight)
            %select_wc 选择最优的弱分类器
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
            T = zeros(1,K); Z = zeros(1,K);
            
            %% 对每一个维度，计算最优的stump参数
            for k = 1:K
                [T(k),Z(k)] = obj.select_stump(points(k,:),labels,weight);
            end
            [~, best.k] = max(abs(Z)); best.t = T(best.k);
            
            %% 迭代寻优二次函数的参数
            F = learn.ssc.DAB_SSC_Pro_Aid(weight,points,labels);
            A = zeros(K); B = zeros(1,K); B(best.k) = 1; C = -best.t;
            x0 = learn.tools.ABC2X(A,B,C);
            % x0 = 0.01*randn(7,1);
            
            %% 使用梯度下降迭代 
%             parameters.learn_rate = 0.1; % 学习速度
%             parameters.momentum = 0.9; % 加速动量
%             parameters.epsilon = 1e-3; % 当梯度的范数小于epsilon时迭代结束
%             parameters.max_it = 1e5; % 最大迭代次数
%             x = learn.optimal.maximize_g(F,x0,parameters);
            
            %% 使用共轭梯度迭代 
            parameters.epsilon = 1e-4; %当梯度模小于epsilon时停止迭代
            parameters.alfa = 1e+3; %线性搜索区间倍数
            parameters.beda = 1e-8; %线性搜索的停止条件
            parameters.max_it = 1e4; %最大迭代次数
            parameters.reset = 500; %重置条件
            [x1,z1] = learn.optimal.maximize_cg(F,x0,parameters);
            [x2,z2] = learn.optimal.minimize_cg(F,x0,parameters);
            if abs(z1) > abs(z2), x = x1; else x = x2; end
            wc = learn.ssc.QuadraticSSC();
            [wc.A,wc.B,wc.C] = learn.tools.X2ABC(x); wc.a = 2; wc.b = -1; 
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
                A = obj.weak{m}.A; B = obj.weak{m}.B; C = obj.weak{m}.C;
                f = learn.tools.quadratic(A,B,C,points);
                c(m,:) = f > 0;
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
            
            %% 计算所有弱分类器的输出
            for m=1:M
                f(m,:) = obj.weak{m}.compute(points,pairs);
            end
            
            %% 累加弱分类器的输出并判决
            F = obj.alfa * f;
            y = F > 0;
        end
        
        %% 训练
        function obj = train(obj,points,labels,M)
            % train 训练DiscreteAdaBoostSSCPro模型
            % 输入：
            % points 数据点
            % labels 标签，共有3行(第1/2行是数据点下标，第3行是相似标签+1或-1,每1列表示一个样本对)
            % M 弱分类器的个数
            % 输出：
            % obj 经过训练的DiscreteAdaBoostSSCPro对象
            
            %% 初始化
            [~,N] = size(points); % 得到数据点数目
            [~,Q] = size(labels); % 得到数据对数目
            I = labels(1,:); J = labels(2,:); L = labels(3,:); P = [I;J];
            Fx = zeros(1,Q); % 用来记录函数Fx的值
            weight = ones(1,Q) / Q; % 初始化权值
            
            %% 迭代
            for m = 1:M
                %% 选择最优的弱分类器
                % 弱分类器fm是一个stump函数，由4个参数确定：(a,b,k,t)
                % fm = a * [(f(x1) > t)==(f(x2) > t)] + b
                wc = obj.select_wc(points,labels,weight);
                
                %% 将最优弱分类器加入到强分类器中
                obj.weak{1+length(obj.weak)} = wc;
                
                %% 计算弱分类器的输出，计算加权值并更新强分类器的函数值
                fm = wc.compute(points,P); % 计算弱分类器的输出
                err = weight * xor((fm > 0),(L > 0))';
                err = max(err,1e-100);
                c = 0.5 * log((1 - err)/err);
                obj.alfa = [obj.alfa c];
                Fx = Fx + c.*fm; % 更新强分类器
                
                %% 更新并归一化权值
                weight = weight .* exp(-c.*L.*fm); % 更新权值
                weight = weight ./ sum(weight); % 归一化权值
                
                %% 计算错误率
                disp(sum(xor(Fx>0,L>0)) / Q);
                
                %% 画图
                A = wc.A; B = wc.B; C = wc.C;
                f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
                warning('off','MATLAB:ezplotfeval:NotVectorized');
                % ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
                ezplot(f,[-10,10,-10,10]);
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
            
            ssc = learn.ssc.DiscreteAdaBoostSSCPro();
            
            N = 1000;
            [points,labels] = learn.data.GenerateData.type7(N);
            plot(points(1,:),points(2,:),'.'); hold on;
            
            M = 30;
            %ssc = ssc.train(points,labels,M);
            %save('ssc.mat','ssc');
            
            load('ssc.mat');
            
            M = 1e6;
            all_points = 20 * rand(2,M) - repmat([10 10]',1,M);
            code = ssc.hash(all_points);
            query = [0,6]';
            query_code = ssc.hash(query);
            z = ones(size(code)); 
            diff = xor(code,repmat(query_code,1,M)); 
            z(diff) = -1;
            dis = ssc.alfa * z;
            idx = (dis + 0.425) > 0;
            figure(2);
            plot(all_points(1,idx),all_points(2,idx),'+'); hold on;
            plot(query(1),query(2),'*r'); 
        end
    end
end

