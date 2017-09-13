function [a,b]=AR(F,x0,h0)
% AR advance & retreat 进退法确定搜索区间
% 输入：
%   F 函数对象，调用F.object(x)计算目标函数
%   x0 起始搜索位置
%   h0 起始搜索步长, h0 > 0
% 输出：
%   a 搜索区间左端点
%   b 搜索区间右端点

    lamda = 1.0;
    x1 = x0;              F1 = F.object(x1); 
    x2 = x0 + lamda * h0; F2 = F.object(x2);
    if F1 > F2
        x = x2; Fx = F2;
        while true
            lamda = 2 * lamda; % 扩大步长
            x1 = x2; 
            x2 = x1 + lamda * h0; 
            F2 = F.object(x2);
            if F2 > Fx
                break;
            end
        end
    else
        x = x1; Fx = F1;
        while true
            lamda = 2 * lamda; % 扩大步长
            x1 = x2; 
            x2 = x1 - lamda * h0; 
            F2 = F.object(x2);
            if F2 > Fx
                break;
            end
        end
    end
    
    a=min(x,x2);
    b=max(x,x2);
end