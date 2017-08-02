function lamda = GoldenSection(F,x,d,a,b,epsilon)
%GoldenSection 此处显示有关此函数的摘要
%   此处显示详细说明
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
        lamda = optimal.GoldenSection(F,x,d,a0,min(alfa0,lamda),epsilon);
    end
end

