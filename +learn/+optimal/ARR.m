function [a,b]=ARR(F,x0,h0,parameters)
% ARR 改进的进退法确定射线搜索区间
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数
%   x0 起始搜索位置
%   h0 起始搜索步长，h0的正负代表了搜索方向
% 输出：
%   a 搜索区间左端点
%   b 搜索区间右端点

    %% 参数设置
    if nargin <= 3 % 没有给出参数的情况
        parameters = [];
        % disp('调用gold函数时没有给出参数集，将使用默认参数');
    end
    
    if ~isfield(parameters,'epsilon') % 给出参数但是没有给出epsilon的情况
        parameters.epsilon = 1e-6; 
        % disp(sprintf('调用ARR函数时参数集中没有epsilon参数，将使用默认值%f',parameters.epsilon));
    end

    %%
    F0 = F.object(x0); 
    x1 = x0 + h0; F1 = F.object(x1);
    
    if F0 > F1
        while F0 > F1
            h0 = 2 * h0; % 扩大步长
            x1 = x0 + h0; F1 = F.object(x1);
        end
        a = x0;
        b = x1;
    else
        a = x0;
        while F0 <= F1
            h0 = 0.5 * h0; % 缩小步长
            if abs(h0) < parameters.epsilon
                break;
            end
            b  = x1;
            x1 = x0 + h0; F1 = F.object(x1);
        end
    end
end