%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [vhW,vb,hb] = GaussianRBM(batchdata,params)

[n, d, nBatches] = size(batchdata);

ob_window_size = nBatches;     % 设定观察窗口的大小为
ob_var_num = 1;                     % 设定观察变量的个数
ob = learn.Observer('重建误差',ob_var_num,ob_window_size,'xxx'); %初始化观察者，观察重建误差
ob = ob.initialize(20.1961);

assert(params.v_var > 0);
fstd = ones(1,d)*sqrt(params.v_var);
params.v_var=[];

r = params.epsilonw_vng;

assert( all(size(params.PreWts.vhW) == [d params.nHidNodes]));
assert( all(size(params.PreWts.hb) == [1 params.nHidNodes]));
assert( all(size(params.PreWts.vb) == [1 d]));

vhW = params.PreWts.vhW;
vb = params.PreWts.vb;
hb = params.PreWts.hb;

vhWInc = zeros(d, params.nHidNodes);
hbInc  = zeros(1, params.nHidNodes);
vbInc  = zeros(1, d);

Ones = ones(n,1);

for it = 0:params.max_it
        batch = mod(it,nBatches)+1;
    
        % POSITIVE PHASE
        data = batchdata(:,:,batch); %nxd
        pos_hidprobs = 1./(1+exp(-data*vhW - Ones*hb)); % p(h_j = 1|data)
        pos_hidstates = pos_hidprobs > rand(size(pos_hidprobs));
    
        pos_prods = data'*pos_hidprobs;
        pos_hid_act = sum(pos_hidprobs);
        pos_vis_act = sum(data); % see notes
        % END POSITIVE PHASE
        
        for iterCD = 1:params.nCD
            
            % NEGATIVE PHASE
            negdataprobs = pos_hidstates*vhW'+Ones*vb;
            negdata = negdataprobs + randn(n,d);
            neg_hidprobs = 1./(1+exp(-negdata*vhW - Ones*hb)); % updating hidden nodes again
            pos_hidstates = neg_hidprobs > rand(size(neg_hidprobs));
        end % end CD iterations
        
        neg_prods = negdata'*neg_hidprobs;
        neg_hid_act = sum(neg_hidprobs);
        neg_vis_act = sum(negdata); % see notes
        
        % END OF NEGATIVE PHASE
        
        r_error_new = sum(sum((data-negdata).^2)) / n;
        message = strcat('重建误差：',num2str(r_error_new));
        %disp(message);
        ob = ob.showit(r_error_new,message);
        
        if it > params.init_final_momen_iter
            momentum = params.final_momen;
        else
            momentum = params.init_momen;
        end
        
        % UPDATE WEIGHTS AND BIASES
        
        vhWInc = momentum * vhWInc + (r/n) * (pos_prods-neg_prods);
        vbInc = momentum * vbInc + (r/n) * (pos_vis_act - neg_vis_act);
        hbInc = momentum * hbInc + (r/n) * (pos_hid_act - neg_hid_act);
        
        vhW = vhW + vhWInc;
        vb = vb + vbInc;
        hb = hb + hbInc;
   
        % END UPDATES
end
        