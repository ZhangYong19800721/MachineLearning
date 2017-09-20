classdef whiten
    %WHITEN ZCA Whitening Transform
    %   参考“Learning Multiple Layers of Features from Tiny Images”的Appendix A
    properties
        W;  % 白化矩阵
        D;  % 解白化矩阵
        A;  % 均值向量
    end

    methods
        function obj = pca(obj,X)
            [~,N] = size(X);
            obj.A = mean(X,2);
            X = double(X) - repmat(obj.A,1,N);
            [E,S] = eig(X*X'/N); d = diag(S);
            [d,i] = sort(d,'descend'); d(d<1e-10) = 0;
            dw = d; dw(d>0) = dw(d>0).^-0.5;
            dd = d; dd(d>0) = dd(d>0).^+0.5;
            E = E(:,i);
            obj.W = diag(dw) * E';
            obj.D = E * diag(dd);
        end
        
        function Y = white(obj,X)
            Y = X - repmat(obj.A,1,size(X,2));
            Y = obj.W * Y;
        end
        
        function X = dewhite(obj,Y)
            X = obj.D * Y;
            X = X + repmat(obj.A,1,size(X,2));
        end
    end
    
    methods(Static)
        function [] = unit_test()
            clear all
            close all
            
            [data,~,~,~] = learn.data.import_mnist('./+learn/+data/mnist.mat');
            data = reshape(data,784,[]);
            whiten = learn.tools.whiten();
            [Y,V,E,D] = whiten.pca_temp(data);
            whiten = whiten.pca(data);
            
            w_data = whiten.white(data);
            zz = w_data * w_data';
            xave = mean(w_data,2);
            xstd = std(w_data,0,2);
            
            X = data; A = mean(X,1);
            Y = X - repmat(A,size(X,1),1);
            Y = V * Y;
            
            yave = mean(Y,2);
            ystd = std(Y,0,2);
            
            d_data = whiten.dewhite(w_data);
        end
    end
end

