classdef RealAdaBoost
    %RealAdaBoost 实现Real AdaBoost算法
    %  参考论文"Additve Logistic Regression a Statistical View of Boosting"
    %  这篇论文非常重要，揭示了boost算法和加性统计模型在本质上是相同的，分析
    %  了RealAdaBoost和GentleBoost算法。
    
    properties
        weak; % 包含若干个弱分类器的cell数组
    end
    
    methods
        function obj = RealAdaBoost()
        end
    end
    
    methods(Access = public)
        function y = predict(obj,points)  
            % PREDICT 使用经过训练的模型判断数据点的分类，正例为1，反例为0
            %
            H = length(obj.weak); [~,N] = size(points); % H弱分类器的个数，N数据点数
            f = zeros(H,N);                             % 存储弱分类器的计算结果
            for h=1:H,f(h,:) = obj.weak{h}.predict(points);end
            y = sum(f) > 0;
        end
        
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
            
            f = wc.predict(points); % 计算加权激活概率
            obj.weak{1+length(obj.weak)} = wc;
            w = weight .* exp(-labels .* f);
            w = w ./ sum(w); 
        end
    end
    
    methods(Static)
        function boost = unit_test()
            clear all;
            close all;
            rng(1)
            
            boost = learn.RealAdaBoost();
            
            N = 1e4; epsilon = 1/N;
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
                z_min = inf; best_w = []; best_b = [];
                wc = learn.Weak_LineR();
                
                for i = 1:length(X)
                    for j = 1:length(A)
                        v = [tan(A(j)) 1 -(tan(A(j))*X(i)+Y(i))];
                        wc.w = [v(1) v(2)]; wc.b = v(3);
                        p = wc.predict(points);
                        W_POS_1 = sum(weight( l &  p));
                        W_NEG_1 = sum(weight(~l &  p));
                        W_POS_2 = sum(weight( l & ~p));
                        W_NEG_2 = sum(weight(~l & ~p));
                        z = 2 * sum(sqrt(W_POS_1 * W_NEG_1) + sqrt(W_POS_2 * W_NEG_2));
                        
                        if z < z_min
                            best_w = wc.w;
                            best_b = wc.b;
                            best_x = 0.5 * log((W_POS_1 + epsilon) / (W_NEG_1 + epsilon));
                            best_y = 0.5 * log((W_POS_2 + epsilon) / (W_NEG_2 + epsilon));
                            z_min = z;
                        end
                    end
                end
                
                wc.w = best_w; wc.b = best_b; wc.x = best_x; wc.y = best_y;
                [boost,weight] = boost.ensemble(points,labels,weight,wc);
                
                a = boost.weak{t}.w(1);
                b = boost.weak{t}.w(2);
                q = boost.weak{t}.b;
                myfunc = @(x,y) a*x + b*y + q;
                ezplot(myfunc,[-15,15,-8,8]);
            end
            
%             for t = 1:T
%                 a = boost.weak{t}.w(1);
%                 b = boost.weak{t}.w(2);
%                 q = boost.weak{t}.b;
%                 myfunc = @(x,y) a*x + b*y + q;
%                 ezplot(myfunc,[-15,15,-8,8]);
%             end
            
            c = boost.predict(points);
            error = sum(xor(c,l)) / N
        end
    end
end

