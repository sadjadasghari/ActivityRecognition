
clear all
%% Parameter Definition
% MIMO dynamic system generation. Based on regressor model. Poles are in a
% fixed pool.
% data[DxTxN] - synthetic data for N samples of dimension x frames
% noise[DxTxN] - predefined white noise
% label - labels for each sample

% num_sys = 2; % number of systems
% sys_ord = [2 3]; % order for each system, minimum 2
num_poles = 100; % number of paird conjugate poles
num_frame = 50; % time length for each sample
num_sample = 100; % number of samples per system
dim = 1;

num_Hcol = 10; % hankel colomn number, only used for verification
if num_Hcol <= dim
    num_Hcol = dim + 1;
end

max_order =4;
num_sys = 10;%0;


sys_par = {};
label = ones(1,num_sys*num_sample);


%% Action Generation
% %real poles
% rmin = 0.02;
% rmax = 1;
% rpos = rmin + (rmax-rmin)*rand(num_poles/2,1);
% rneg = -(rmin + (rmax-rmin)*rand(num_poles/2,1));
% rpole = [rneg; rpos];
% for r = 1:length(rpole)
% poles = rpole(r);
% coef = -fliplr(poly(poles'));
%     dic_par{r} = coef;
% end

%complex poles
rmin = 0.02;
rmax = 1;
theta = (rand(num_poles,1))*pi;
rpole = sqrt(rmin^2 + (rmax^2-rmin^2)*rand(num_poles,1));
for r = 1:length(rpole)
    poles = [rpole(r).*exp(theta(r)*1i),...
        rpole(r).*exp(theta(r)*(-1i))]';
    
    poles = poles(:);
    coef = -fliplr(poly(poles'));
    dic_par{r} = coef;
end


sys_orders= randi(max_order,num_sys,1);
for ord_i= 1:max_order
    [tmpsysind,~] = find(sys_orders == ord_i);
    idxpool = nchoosek(1:num_poles,ord_i);
    idxidxpool = randperm(size(idxpool,1));
    idxidxpool  = idxidxpool(1:length(tmpsysind));
    for i = 1:length(tmpsysind)
        p = idxidxpool(i);
        tmpidx = idxpool(p,:);
        
        %     %real poles
        %     poles = rpole(tmpidx);
        
        %complex poles
        poles = [rpole(tmpidx).*exp(theta(tmpidx)*1i),...
            rpole(tmpidx).*exp(theta(tmpidx)*(-1i))]';
        
        poles = poles(:);
        
        sys_ind = tmpsysind(i);
        pole_set{sys_ind} = idxpool(p,:);
        
        %     %real
        %     sys_ord(sys_ind) = sys_orders(sys_ind);
        
        %complex
        sys_ord(sys_ind) = sys_orders(sys_ind)*2;
        
        null{sys_ind} = -fliplr(poly([poles; rand(num_Hcol-1-sys_ord(sys_ind),1)]')); % null space for each system, depends on the number of hankel colomns
        coef = -fliplr(poly(poles'));
        sys_par{sys_ind} = coef;
    end
end
%% Data Generation
noise_level=0.1;
noise_data = randn(dim,num_frame,num_sys*num_sample);
noiselevel_data = noise_data;
data = zeros(dim,num_frame,num_sys*num_sample);

for k=1:num_sys
    temp_data = zeros(dim,num_frame,num_sample);
    initial_value = [];
    for i = 1:num_sample
        initial_value(:,:,i) = rand(dim,sys_ord(k))-0.5;
        output_value(:,1:sys_ord(k)) = initial_value(:,:,i);
        for j = sys_ord(k)+1:num_frame
            %             output_value(:,j) = sys_par{k}(1:end-1)*output_value(j-sys_ord(k):j-1,1);
            output_value(:,j) = output_value(:,j-sys_ord(k):j-1)*sys_par{k}(1:end-1)';
        end
        %         temp_value = (output_value-mean(output_value(:)))/std(output_value(:));
        temp_value = output_value/max(abs(output_value(:)));
        temp_data(:,:,i) = temp_value;
    end
    data(:,:,num_sample*(k-1)+1:num_sample*k) = temp_data;
    label((k-1)*num_sample+1:k*num_sample) = ones(num_sample,1)*k;
end


% verify with null space vector
for s = 1:num_sys
    idx = randi(num_sample,1);
    test = data(:,:,(s-1)*num_sample+idx);
    H = hankel_mo(test,[dim*(num_frame-num_Hcol+1),num_Hcol]);
    fprintf('System %d --- ||wH||_2 = %0.1f\n',s,norm(H*null{s}',2));
end

data_noise = data + noiselevel_data;

% %% Generate random projection parameters
% temp_randpro = randn(num_MAXrandpro,num_Hcol);
% randpro = temp_randpro./repmat(sum(temp_randpro.^2,2),1,num_Hcol);
% temp_randpro = randn(num_MAXrandpro,num_Hcol*2);
% randpro_2 = temp_randpro./repmat(sum(temp_randpro.^2,2),1,num_Hcol*2);
% %% Generate random index
% temp_index = randperm(num_sample)';
% for i = 1:num_fold
%     index_train(:,i) = temp_index(unit_train*(i-1)+1:unit_train*i);
%     index_test(:,i) = [temp_index(1:unit_train*(i-1));temp_index(unit_train*i+1:end)];
% end






