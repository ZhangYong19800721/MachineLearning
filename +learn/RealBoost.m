classdef RealBoost
    %RealBoost 实现Real Ada Boost算法
    %  参考论文"Additve Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了RealAdaBoost和GentleBoost算法。
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods
        function obj = RealBoost()
        end
    end
    
    methods(Access = private)
        function [obj,w]  = ensemble(obj,points,labels,weight,wc)
            %ensemble 组合弱分类器，找到一个最优的弱分类器，然后通过此函数组合起来
            %   输入：
            %   points 表示数据点
            %   labels 标签+1或-1
            %   weight 概率分布权值
            %   wc 新加入的弱分类器
            % 
            %   输出：
            %   obj 训练后的adaboost对象
            %   w 更新后的权值
            
            f = wc.compute(points); % 计算弱分类器函数值f(x)
            obj.weak{1+length(obj.weak)} = wc; % 将弱分类器加入组中
            w = weight .* exp(-labels .* f); % 更新权值
            w = w ./ sum(w); % 归一化权值
        end
        
        function wc = find_wc(obj,points,labels,weight)
            % 选择最优的弱分类器
            wc = learn.Weak_LineR(); l = labels > 0;
            [~,N] = size(points); epsilon = 1/N;
            
            X = -15:15; Y = -15:15; A = linspace(-pi/2,pi/2,180); 
            zm = inf;
            for x = X 
                for y = Y
                    for a = A
                        v = [-tan(a) 1 tan(a)*x-y];
                        wc.w = [v(1) v(2)]; wc.b = v(3);
                        p = wc.predict(points);
                        W_POS_1 = sum(weight( l &  p));
                        W_NEG_1 = sum(weight(~l &  p));
                        W_POS_2 = sum(weight( l & ~p));
                        W_NEG_2 = sum(weight(~l & ~p));
                        z = 2 * sum(sqrt(W_POS_1 * W_NEG_1) + sqrt(W_POS_2 * W_NEG_2));
                        if z < zm
                            best.w = wc.w;
                            best.b = wc.b;
                            best.x = 0.5 * log((W_POS_1 + epsilon) / (W_NEG_1 + epsilon));
                            best.y = 0.5 * log((W_POS_2 + epsilon) / (W_NEG_2 + epsilon));
                            zm = z;
                        end
                    end
                end
            end
            
            wc.w = best.w;
            wc.b = best.b;
            wc.x = best.x;
            wc.y = best.y;
        end
    end
    
    methods(Access = public)
        function y = compute(obj,points)
            % compute 计算boost模型的响应函数F(x)
            M = length(obj.weak); [~,N] = size(points); % M弱分类器的个数，N数据点数
            f = zeros(M,N);                             % 存储弱分类器的计算结果
            for m=1:M,f(m,:) = obj.weak{m}.compute(points);end
            y = sum(f,1);
        end
        
        function y = predict(obj,points)  
            % PREDICT 使用经过训练的模型判断数据点的分类，正例为1，反例为0
            y = obj.compute(points) > 0;
        end
        
        function obj = train(obj,points,labels,M)
            % train 训练RealBoost模型
            % 输入：
            % points 数据点
            % labels 标签，+1或-1
            % M 弱分类器的个数
            % 输出：
            % obj 经过训练的boost对象
            
            [~,N] = size(points); % 得到数据点数目
            weight = ones(1,N) / N; % 初始化权值
            for m = 1:M
                wc = obj.find_wc(points,labels,weight);
                [obj,weight] = obj.ensemble(points,labels,weight,wc);
                
                l = labels > 0;
                y = obj.predict(points);
                error(m) = sum(xor(y,l)) / N
                
                a = obj.weak{m}.w(1);
                b = obj.weak{m}.w(2);
                c = obj.weak{m}.b;
                myfunc = @(x,y) a*x + b*y + c;
                ezplot(myfunc,[-15,15,-8,8]);
                drawnow
            end
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.RealBoost();
            
            N = 1e3;
            [points,labels] = learn.GenerateData.type1(N); l = labels > 0;
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'*'); 
            
            M = 15;
            boost = boost.train(points,labels,M);
            
            y = boost.predict(points);
            error = sum(xor(y,l)) / N
        end
    end
end

