function [ny,nx] = parabola(F,a,b,parameters)
%PARABOLA 使用抛物线法进行精确一维线搜索
%  参考 马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社） 
%  输入：
%       F 单变量函数，F.object(x)计算目标函数在x处的值
%       a 搜索区间的左端点
%       b 搜索区间的右端点
%       parameters.parabola.epsilon 停止条件，典型值1e-6
%  输出：
%       ny 最小点的函数值
%       nx 最小点的变量值
    
    %% 找到满足条件的起始点x1,x2,x3,应满足f(x2)<f(x1)且f(x2)<f(x3)
    x1 = a;            x3 = b;
    f1 = F.object(x1); f3 = F.object(x3);
    % 从中间出发向x1逐步靠近，知道找到比f1和f3都小的函数值f2和点x2
    n = 0; 
    while 1/(2^n) > parameters.parabola.epsilon
        n = n + 1;
        x2 = x1 + (x3 - x1) / 2^n;
        f2 = F.object(x2);
        if f1 > f2 && f2 < f3
            break;
        end
    end
    
    %% 如果找不到满足条件的x2
    if f2 > f1 || f2 > f3
        if f1 < f3
            nx = x1; ny = f1;
        else
            nx = x3; ny = f3;
        end
        return;
    end
    
    %% 开始搜索
    while min(x3-x1,x2-x1) > parameters.parabola.epsilon
        %% 计算插值二次函数的最优点
        alfa = (x2^2 - x3^2)*f1 + (x3^2 - x1^2)*f2 + (x1^2 - x2^2)*f3;
        beda = (x2^1 - x3^1)*f1 + (x3^1 - x1^1)*f2 + (x1^1 - x2^1)*f3;
        if beda == 0
            break;
        else
            xp = 0.5 * alfa / beda;
            if xp <= x1 || xp >= x3
                break;
            end
        end
        
        %% 区间缩小，根据xp与x2，fp与f2的相互关系分六种情况进行区间缩小
        if xp == x2
            state = 7; fp = f2;
        else
            fp = F.object(xp); 
            if xp > x2
                if fp < f2
                    state = 1;
                elseif fp > f2
                    state = 2;
                else
                    state = 3;
                end
            elseif xp < x2
                if fp < f2
                    state = 4;
                elseif fp > f2
                    state = 5;
                else
                    state = 6;
                end
            end
        end
        
        switch state
            case 1
                x1 = x2; x2 = xp;
                f1 = f2; f2 = fp;
            case 2
                x3 = xp;
                f3 = fp;
            case 3
                x1 = x2; x3 = xp; x2 = (x1 + x3)/2; 
                f1 = f2; f3 = fp; f2 = F.object(x2);
            case 4
                x3 = x2; x2 = xp; 
                f3 = f2; f2 = fp; 
            case 5
                x1 = xp;
                f1 = fp;
            case 6
                x1 = xp; x3 = x2; x2 = (x1 + x3)/2; 
                f1 = fp; f3 = f2; f2 = F.object(x2);
            case 7
                x12 = (x2 + x1) / 2; f12 = F.object(x12);
                x23 = (x2 + x3) / 2; f23 = F.object(x23);
                if f12 <= min(f2,f23)
                    x3 = x2 ; f3 = f2; 
                    x2 = x12; f2 = f12;
                elseif f23 <= min(f2,f12)
                    x1 = x2 ; f1 = f2;
                    x2 = x23; f2 = f23;
                elseif f2 <= min(f12,f23)
                    x1 = x12; f1 = f12;
                    x3 = x23; f3 = f23;
                end
        end
    end
    
    ny = f2;
    nx = x2;
end

