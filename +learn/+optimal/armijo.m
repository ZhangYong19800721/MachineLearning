function [ny,nx] = armijo(F,x,g,d,parameters)
%ARMIJO �Ǿ�ȷ��������Armijo׼��
%   �ο� �������Ż����㷽������MATLAB����ʵ�֡���������ҵ�����磩
%   ���룺
%       F ��������F.object(x)����Ŀ�꺯����x����ֵ
%       x ��������ʼλ��
%       g Ŀ�꺯����x�����ݶ�
%       d ��������
%       parameters ������
%   �����
%       ny ���ŵ�ĺ���ֵ
%       nx ���ŵ�ı���ֵ

    %% ��������
    if nargin <= 4 % û�и������������
        parameters = [];
        % disp('����armijo����ʱû�и�������������ʹ��Ĭ�ϲ���');
    end
    
    if ~isfield(parameters,'beda') % ������������û�и���beda�����
        parameters.beda = 0.5; 
        % disp(sprintf('����armijo����ʱ��������û��beda��������ʹ��Ĭ��ֵ%f',parameters.beda));
    end
    
    if ~isfield(parameters,'alfa') % ������������û�и���alfa�����
        parameters.alfa = 0.2; 
        % disp(sprintf('����armijo����ʱ��������û��alfa��������ʹ��Ĭ��ֵ%f',parameters.alfa));
    end
    
    if ~isfield(parameters,'maxs') % ������������û�и���maxs�����
        parameters.maxs = 30;
        % disp(sprintf('����armijo����ʱ��������û��maxs��������ʹ��Ĭ��ֵ%f',parameters.maxs));
    end

    assert(0 < parameters.beda && parameters.beda <   1);
    assert(0 < parameters.alfa && parameters.alfa < 0.5);
    assert(0 < parameters.maxs);
    
    %%
    d = 1e3 * d; % ���������ֵ����1000�����൱�ڿ��ܵ����ѧϰ�ٶ�Ϊ1000
    m = 0; f = F.object(x);
    while m <= parameters.maxs
        nx = x + parameters.beda^m * d;
        ny = F.object(nx);
        if ny <= f + parameters.alfa * parameters.beda^m * g'* d
            break;
        end
        m = m + 1;
    end
end

