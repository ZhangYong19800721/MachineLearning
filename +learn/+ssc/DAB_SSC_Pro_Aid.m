classdef DAB_SSC_Pro_Aid
    %DAB_SSC_Pro_Aid DiscreteAdaBoostSSCPro迭代优化的辅助类
    %   此处显示详细说明
    
    properties
        weight;
        points;
        labels;
    end
    
    methods
        function obj = DAB_SSC_Pro_Aid(weight,points,labels)
            obj.weight = weight;
            obj.points = points;
            obj.labels = labels;
        end
        
        function g = gradient(obj,x)
            %% 初始化
            [K,N] = size(obj.points); Q = K*K+K+1; % K数据的维度、N数据点数、Q二次方程参数个数
            [A,B,C] = learn.tools.X2ABC(x); % 将x分解为A、B、C三个参数
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% 计算梯度
            f = learn.tools.quadratic(A,B,C,obj.points); % 计算所有点的f函数值
            h = learn.tools.sigmoid(f); % 计算所有点的h函数值
            g_h_C = ones(1,N); % h函数对C的梯度
            g_h_B = obj.points; % h函数对B的梯度
            g_h_A = 0.5 * obj.points(reshape(repmat(1:K,K,1),1,[]),:) .* obj.points(repmat(1:K,1,K),:); % h函数对A的梯度
            g_h_x = repmat(h.*(1-h),Q,1) .* [g_h_A;g_h_B;g_h_C];
            g_c_x = 4 * (g_h_x(:,I) .* (g_h_x(:,J) - 0.5)) + 4 * (g_h_x(:,J) .* (g_h_x(:,I) - 0.5));
            g = sum(repmat(obj.weight.*L,Q,1) .* g_c_x,2);
        end
        
        function y = object(obj,x)
            %% 初始化
            [K,N] = size(obj.points);
            [A,B,C] = learn.tools.X2ABC(x); % 将x分解为A、B、C三个参数
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% 计算目标函数值
            f = learn.tools.quadratic(A,B,C,obj.points); % 计算所有点的f函数值
            h = learn.tools.sigmoid(f); % 计算所有点的h函数值
            c = 4 * (h(I) - 0.5) .* (h(J) - 0.5); % 计算所有点的c函数值
            y = sum(obj.weight .* L .* c,2); % 计算目标函数值
        end
    end
    
    methods(Static)
        function error_idx = unit_test()
            rng(2);
            [points,labels] = learn.data.GenerateData.type10(); [~,Q] = size(labels);
            weight = ones(1,Q) / Q;
            aid = learn.ssc.DAB_SSC_Pro_Aid(weight,points,labels);
            x0 = [2 0 0 2 0 0 -25]';
            %% 画图
            [A,B,C] = learn.tools.X2ABC(x0); 
            f = @(x,y) 0.5*[x y]*A*[x y]' + B*[x y]' + C;
            warning('off','MATLAB:ezplotfeval:NotVectorized');
            % ezplot(f,[min(points(1,:)),max(points(1,:)),min(points(2,:)),max(points(2,:))]);
            ezplot(f,[-10,10,-10,10]);
            drawnow;
            %%
            y0 = aid.object(x0);
            g0 = aid.gradient(x0);
            error_idx = 0;
        end
    end
    
end

