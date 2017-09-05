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
            [K,N] = size(obj.points);
            A = reshape(x(1:(K*K)),K,K); B = reshape(x(K*K+(1:K)),1,[]); C = x(end);
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% 
            f = 0.5 * sum((obj.points' * A) .* obj.points',2)' + B * obj.points + repmat(C,1,N); % 计算所有点的f函数值
            h = +learn.tools.sigmoid(f); % 计算所有点的h函数值
            g_h_C = ones(1,N); % h函数对C的梯度
            g_h_B = obj.points; % h函数对B的梯度
            g_h_A = zeros(K*K,N);
            for n = 1:N
                g_h_A(:,n) = reshape(0.5 * obj.points(:,n) * obj.points(:,n)',[],1); % h函数对A的梯度
            end
            g_h_x = repmat(h .* (h - 1),K*K+K+1,1) .* [g_h_A;g_h_B;g_h_C];
            g_c_x = 4 * (g_h_x(:,I) .* (g_h_x(:,J) - 0.5)) + 4 * (g_h_x(:,J) .* (g_h_x(:,I) - 0.5));
            g = sum(repmat(obj.weight .* L,K*K+K+1,1) .* g_c_x,2);
        end
        
        function y = object(obj,x)
            %% 初始化
            [K,N] = size(obj.points);
            A = reshape(x(1:(K*K)),K,K); B = reshape(x(K*K+(1:K)),1,[]); C = x(end);
            I = obj.labels(1,:); J = obj.labels(2,:); L = obj.labels(3,:);
            
            %% 计算目标函数值
            f = 0.5 * sum((obj.points' * A) .* obj.points',2)' + B * obj.points + repmat(C,1,N); % 计算所有点的f函数值
            h = learn.tools.sigmoid(f); % 计算所有点的h函数值
            c = 4 * (h(I) - 0.5) .* (h(J) - 0.5); % 计算所有点的c函数值
            y = sum(obj.weight .* L .* c,2); % 计算目标函数值
        end
    end
    
end

