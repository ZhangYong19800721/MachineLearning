function [x,y] = minimize_dfp(F,x0,parameters)
%MINIMIZE_DFP DFP��ţ�ٷ������Լ�����Ż�����
%   �ο� �������Ż����㷽������MATLAB����ʵ�֡���������ҵ�����磩 

    %% ��������
    if nargin <= 2 % û�и�������
        parameters = [];
        disp('����minimize_dfp����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-5; 
        disp(sprintf('����minimize_dfp����ʱ��������û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('����minimize_dfp����ʱ��������û��max_it��������ʹ��Ĭ��ֵ%f',parameters.max_it));
    end
    
    if ~isfield(parameters,'armijo') % ������������û�и���armijo
        parameters.armijo = [];
    end

    %%
    x1 = x0;
    D = length(x0); H = eye(D); % D����ά�ȣ�B��Hessen����Ľ����������ʼʱΪ��λ��
    G1 = F.gradient(x1); y1 = F.object(x1); % ����x1�����ݶȺͺ���ֵ
    for it = 0:parameters.max_it
        ng1 = norm(G1); % �����ݶ�ģ
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f',y1,it,ng1));
        if ng1 < parameters.epsilon, break; end   %���ݶ�ģ�㹻Сʱ��ֹ����
        d = -H * G1;  % ������������
        [y2,x2] = learn.optimal.armijo(F,x1,G1,d,parameters.armijo);
        G2 = F.gradient(x2);  %����x2�����ݶ�
        s = x2 - x1;
        z = G2 - G1;
        if z' * s > 0
            H = H - (H*(z*z')*H)/(z'*H*z)+(s*s')/(s'*z);
        end
        x1 = x2; G1 = G2; y1 = y2;
    end
    x = x1;
    y = y1;
end

