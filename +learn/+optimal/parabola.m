function [ny,nx] = parabola(F,a,b,parameters)
%PARABOLA 使用抛物线法进行精确线搜索
%  参考 马昌凤“最优化计算方法及其MATLAB程序实现”（国防工业出版社） 
%  输入：
%       F 单变量函数，F.object(x)计算目标函数在x处的值
%       a 搜索区间的左端点
%       b 搜索区间的右端点
%       parameters.parabola.epsilon 停止条件，典型值1e-6
%  输出：
%       ny 最小点的函数值
%       nx 最小点的变量值

    s0 = x;            s1 = s0 + h;       s2 = s0 + 2*h;
    F0 = F.object(s0); F1 = F.object(s1); F2 = F.object(s2);
    
    h = (3*F0 - 4*F1 + F2) * h / (2*(2*F1 - F0 - F2));
    s0 = s0 + h;
end

