classdef whiten
    %WHITEN ZCA Whitening Transform
    %   �ο���Learning Multiple Layers of Features from Tiny Images����Appendix A
    properties
        W;  % �׻�����
        D;  % ��׻�����
        A;  % ��ֵ����
    end

    methods
        function obj = pca_temp(obj,X)
            % do PCA on image patches
            % INPUT variables:
            % X                  matrix with image patches as columns
            % OUTPUT variables:
            % Y                  the project matrix of the input data X without whiting
            % V                  whitening matrix
            % E                  principal component transformation (orthogonal)
            % D                  variances of the principal components
            
            
            X = X-ones(size(X,1),1)*mean(X); %ȥ��ֱ���ɷ�
            covarianceMatrix = X*X'/size(X,2); % �����Э�������
            [E,D] = eig(covarianceMatrix); % E�������������ɣ�����ÿһ��������������D������ֵ���ɵĶԽǾ���
                                            % ��Щ����ֵ������������û�о�������
            % Sort the eigenvalues  and recompute matrices
            % ��Ϊsort�������������У�����Ҫ���ǽ������У�������ȡ����,diag(a)��ȡ��a�ĶԽ�Ԫ�ع���
            % һ���������������dummy�ǽ������к��������order��������˳��
            [~,order] = sort(diag(D),'descend');
            E = E(:,order);%������������������ֵ��С���н������У�ÿһ����һ����������
            Y = E'*X;
            d = diag(D); %d��һ��������
            %dsqrtinv��������������ֵ�����ź�ȡ��,��Ȼ��������ֵ�йص�������
            %��ʵ�����󿪸��ź�������
            dsqrtinv = real(d.^(-0.5));
            Dsqrtinv = diag(dsqrtinv(order));%��һ���ԽǾ��󣬾����е�Ԫ��ʱ���������к��˵�����ֵ������ȡ���ŵ���
            D = diag(d(order));%D��һ���ԽǾ�����Խ�Ԫ��������ֵ�Ӵ�С����
            V = Dsqrtinv*E';%����ֵ�������������������
        end
        
        function obj = pca(obj,X)
            [D,N] = size(X);
            obj.A = mean(X,1);
            X = double(X) - repmat(obj.A,D,1);
            [V,S] = eig(X*X');
            obj.W =     sqrt(N-1) * V * diag(diag(S).^-0.5) * V';
            obj.D = 1 / sqrt(N-1) * V * diag(diag(S).^+0.5) * V';
        end
        
        function Y = white(obj,X)
            Y = X - repmat(obj.A,size(X,1),1);
            Y = obj.W * Y;
        end
        
        function X = dewhite(obj,Y)
            X = obj.D * Y;
            X = X + repmat(obj.A,size(X,1),1);
        end
    end
end

