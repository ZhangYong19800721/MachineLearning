function lamda = search(F,x,d,a,b,epsilon)
%search 使用黄金分割法进行线搜索
%   
    g = (sqrt(5)-1)/2;
    a0 = a; alfa0 = a + (1 - g)*(b - a);
    alfa = a + (1 - g)*(b - a);
    beda = a + g * (b - a);
    F_x_ad = F.object(x +    a * d);
    F_alfa = F.object(x + alfa * d);
    F_beda = F.object(x + beda * d);
    
    while b - a > epsilon
        if F_alfa > F_beda
            a = alfa;
            alfa = beda; F_alfa = F_beda;
            beda = a + g * (b - a);
            F_beda = F.object(x + beda * d);
        else
            b = beda;
            beda = alfa; F_beda = F_alfa;
            alfa = a + (1 - g)*(b - a);
            F_alfa = F.object(x + alfa * d);
        end
    end
    
    lamda = (a + b) / 2;
    if F_x_ad < F.object(x+lamda*d)
        lamda = learn.optimal.search(F,x,d,a0,min(alfa0,lamda),epsilon);
    end
end

