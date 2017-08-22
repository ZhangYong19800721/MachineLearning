classdef DiscreteAdaBoost2
    %DiscreteAdaBoost2 实现Discrete AdaBoost2算法
    %  参考论文"Additve Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了RealAdaBoost和GentleBoost算法。
    %  DiscreteAdaBoost2与DiscreteAdaBoost1是等价的
    
    properties
        weak; % 包含若干个弱分类器的cell数组
        alfa; % 每个弱分类器的投票权值
    end
    
    methods
        function obj = DiscreteAdaBoost2()
        end
    end
    
    methods
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
            
            c = wc.predict(points); % c是一个logical变量
            l = labels > 0; % labels也换成logical变量
            r = weight * xor(c,l)';
            beda = log((1 - r)/r);
            obj.weak{1+length(obj.weak)} = wc;
            obj.alfa = [obj.alfa beda];
            w = weight .* exp(beda .* xor(c,l));
            w = w ./ sum(w);
        end
        
        function y = predict(obj,points)  % 使用经过训练的模型判断数据点的分类
            %PREDICT 判断两个点是否相似，相似为+1，不相似为-1
            %
            H = length(obj.alfa); [~,N] = size(points); % H弱分类器的个数，N数据点数
            c = false(H,N); % 存储弱分类器的分类结果
            for h=1:H,c(h,:) = obj.weak{h}.predict(points);end
            k = -ones(size(c)); k(c) = 1;
            y = obj.alfa * k > 0;
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.DiscreteAdaBoost2();
            
            N = 1e4;
            [points,labels] = learn.GenerateData.type1(N); l = labels > 0;
            
            figure;
            group1 = points(:,labels== 1);
            group2 = points(:,labels==-1);
            plot(group1(1,:),group1(2,:),'+'); hold on;
            plot(group2(1,:),group2(2,:),'*'); 
            
            weight = ones(1,N)/N; % 初始化权值
            [X,Y] = meshgrid(-15:15,-15:15); A = (-pi/2+eps):0.1:(pi/2-eps); 
            X = reshape(X,1,[]); Y = reshape(Y,1,[]);
            T = 10;
            
            for t = 1:T
                r_min = inf; best_w = []; best_b = [];
                wc = learn.Weak_Line();
                for i = 1:length(X)
                    for j = 1:length(A)
                        wc.w = [tan(A(j)) 1]; wc.b = -(tan(A(j))*X(i)+Y(i));
                        c = wc.predict(points);
                        r = weight * xor(c,l)';
                        if r < r_min
                            best_w = wc.w;
                            best_b = wc.b;
                            r_min = r;
                        end
                    end
                end
                
                wc.w = best_w; wc.b = best_b;
                [boost,weight] = boost.ensemble(points,labels,weight,wc);
            end
            
            for t = 1:T
                a = boost.weak{t}.w(1);
                b = boost.weak{t}.w(2);
                c = boost.weak{t}.b;
                myfunc = @(x,y) a*x + b*y + c;
                ezplot(myfunc,[-15,15,-8,8]);
            end
            
            c = boost.predict(points);
            error = sum(xor(c,l)) / N;
        end
    end
end

