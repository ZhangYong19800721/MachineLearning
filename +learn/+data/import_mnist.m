function [train_images,train_labels,test_images,test_labels] = import_mnist(file_name)
    load(file_name);
    train_images = mnist_train_images;
    train_labels = mnist_train_labels;
    minibatch_size = 100;
    minibatch_num = 0;
    
    for n = 1:length(train_labels)
        idx = mod(n-1,10);
        if idx == train_labels(n)
            % do nothing!
        else
            flag = false;
            for k = (n+1):length(train_labels)
                if idx == train_labels(k)
                    swap = train_labels(k);
                    train_labels(k) = train_labels(n);
                    train_labels(n) = swap;
                    
                    swap = train_images(:,k);
                    train_images(:,k) = train_images(:,n);
                    train_images(:,n) = swap;
                    
                    flag = true;
                    break;
                end
            end
            
            if flag == false
                minibatch_num = floor((n - 1)/minibatch_size);
                break;
            end
        end
    end
    
    train_images = train_images(:,1:(minibatch_size*minibatch_num));
    train_images = double(reshape(train_images,784,minibatch_size,minibatch_num)) / 255;
    train_labels = train_labels(1:(minibatch_size*minibatch_num));
    train_labels = reshape(train_labels,minibatch_size,minibatch_num);
    
    test_images = double(mnist_test_images) / 255;
    test_labels = mnist_test_labels;
end