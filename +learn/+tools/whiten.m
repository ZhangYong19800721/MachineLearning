classdef whiten
    %WHITEN ZCA Whitening Transform
    %   参考“Learning Multiple Layers of Features from Tiny Images”的Appendix A
    properties
        W;  % 白化矩阵
        D;  % 解白化矩阵
        A;  % 均值向量
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
            
            
            X = X-ones(size(X,1),1)*mean(X); %去除直流成分
            covarianceMatrix = X*X'/size(X,2); % 求出其协方差矩阵
            [E,D] = eig(covarianceMatrix); % E是特征向量构成，它的每一列是特征向量，D是特征值构成的对角矩阵
                                            % 这些特征值和特征向量都没有经过排序
            % Sort the eigenvalues  and recompute matrices
            % 因为sort函数是升序排列，而需要的是降序排列，所以先取负号,diag(a)是取出a的对角元素构成
            % 一个列向量，这里的dummy是降序排列后的向量，order是其排列顺序
            [~,order] = sort(diag(D),'descend');
            E = E(:,order);%将特征向量按照特征值大小进行降序排列，每一列是一个特征向量
            Y = E'*X;
            d = diag(D); %d是一个列向量
            %dsqrtinv是列向量，特征值开根号后取倒,仍然是与特征值有关的列向量
            %其实就是求开根号后的逆矩阵
            dsqrtinv = real(d.^(-0.5));
            Dsqrtinv = diag(dsqrtinv(order));%是一个对角矩阵，矩阵中的元素时按降序排列好了的特征值（经过取根号倒后）
            D = diag(d(order));%D是一个对角矩阵，其对角元素由特征值从大到小构成
            V = Dsqrtinv*E';%特征值矩阵乘以特征向量矩阵
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

