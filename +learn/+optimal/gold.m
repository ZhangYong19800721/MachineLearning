function [lamda,nf,nx] = gold(F,x,d,parameters)
%gold 使用黄金分割法进行线搜索
%   
    a = parameters.gold.a;
    b = parameters.gold.b;
    epsilon = parameters.gold.epsilon;
    
    g = (sqrt(5)-1)/2;
    ax = a + (1 - g)*(b - a); Fax = F.object(x + ax * d);
    bx = a + g * (b - a);     Fbx = F.object(x + bx * d);
    
    while b - a > epsilon
        if Fax > Fbx
            a = ax;
            ax = bx; Fax = Fbx;
            bx = a + g * (b - a);
            Fbx = F.object(x + bx * d);
        else
            b = bx;
            bx = ax; Fbx = Fax;
            ax = a + (1 - g)*(b - a);
            Fax = F.object(x + ax * d);
        end
    end
    
    if Fax > Fbx
        lamda = b; nx = x + lamda * d; nf = Fbx;
    else
        lamda = a; nx = x + lamda * d; nf = Fax;
    end
end

