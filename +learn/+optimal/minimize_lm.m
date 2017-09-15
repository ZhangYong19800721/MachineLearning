function [x1,z1] = minimize_lm(F,x0,parameters)
%minimize_lm LevenbergMarquardt�㷨��һ�������С�����������ֵ�㷨
%   ��һ�ָĽ��ĸ�˹ţ�ٷ�����������������ڸ�˹ţ�ٷ���������½�����֮��
%   ���룺
%       F ����[H,G]=F.hessian(x)����Ŀ�꺯���Ľ���Hessian����H=J'*J,JΪJacobi���󣩺��ݶ�G��
%         ��H��Gһ������������ΪJacobi����ļ����������������޷��洢
%         ����F.object(x) ����Ŀ�꺯����ֵ
%       x0 ��������ʼλ��
%       parameters ������
%   �����
%       x ���ŵĲ�����
%       y ��С�ĺ���ֵ

    %% ��������
    if nargin <= 2 % û�и�������
        parameters = [];
        disp('����minimize_lm����ʱû�и�������������ʹ��Ĭ�ϲ�����');
    end
    
    if ~isfield(parameters,'epsilon') % ������������û�и���epsilon
        parameters.epsilon = 1e-3; 
        disp(sprintf('����minimize_lm����ʱ��������û��epsilon��������ʹ��Ĭ��ֵ%f',parameters.epsilon));
    end
    
    if ~isfield(parameters,'max_it') % ������������û�и���max_it
        parameters.max_it = 1e6;
        disp(sprintf('����minimize_lm����ʱ��������û��max_it��������ʹ��Ĭ��ֵ%d',parameters.max_it));
    end

    %%
    alfa = 0.01; beda = 10; 
    D = length(x0); % ���ݵ�ά��
    x1 = x0;
    
    for it = 1:parameters.max_it
        [H1,G1] = F.hessen(x1);
        F1 = F.object(x1);
        ng1 = norm(G1);
        if ng1 < parameters.epsilon
            break;
        end
        delta = -(H1 + alfa * eye(D)) \ (G1/2);
        x2 = x1 + delta;
        F2 = F.object(x2);
        
        disp(sprintf('Ŀ�꺯��:%f ��������:%d �ݶ�ģ:%f',F1,it,ng1));
        
        if F1 > F2
            alfa = alfa / beda;
            x1 = x2;
        else
            alfa = alfa * beda;
        end
    end
end

