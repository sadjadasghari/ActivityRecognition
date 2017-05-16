% function [ alpha_ind ] = coeff_computation( data, sys_par, num_poles )
% This function returns the set of coefficients for every trajectory sample
% based on the vocabulary of poles.
clear all;
load('100realPoles_10000data_100classes_maxord1.mat');

% It is assumed that the trajectories are single-channel.
[~,T, N] = size(data_noise);
alpha = zeros(N,num_poles);
alpha_ind = zeros(N,num_poles);
% Generate impulse responses for every one-order system from each pole
sys = cell(num_poles,1);
Up = cell(num_poles,1);
y = cell(num_poles,1);
W = zeros(T,num_poles);
for p = 1:num_poles%numel(sys_par)%num_poles
    sys{p} = tf([1,0],[poly(rpole(p))],0.3);
    yy = impulse(sys{p});
    if length(yy)<T
        yy = [yy;zeros(T - length(yy),1)];
    end
    y{p} = yy(1:T,:);
    W(:,p) = y{p};
    Hp = hankel_mo(y{p}');
    [Up{p},~,~] = svd(Hp);
end

% Compute the coefficients for every sample
for i = 1:N
    H = hankel_mo(data_noise(:,:,i));
    [U,S,~] = svd(H);
    r = nnz(diag(S)>1);
%     disp(r);
    for j = 1:numel(Up)
        [~,Sp,~] = svd([U(:,1:r), Up{j}(:,1:max_order)]);
        eigen = diag(Sp);
        var = sum(eigen(1:max_order))/max_order;
        alpha_ind(i,j) = var;
        %alpha_ind(i,j) = var;
%         if var >= 1.4142 && var <= 1.41422 %sqrt(2)
%             alpha_ind(i,j) = 1;
%         else
%             alpha_ind(i,j) = 0;
%         end
        %alpha(i,j) = trace(H'*Hp{j});
        %alpha(i,j) = sum(sum(H'*Hp{j}));
        %alpha(i,j) = sum(dot(H,Hp{j}));
    end
end
%% 

lambda = 0.002;
coeffs = zeros(num_poles,N);
for d = 1:N
kappa = 1;
cvx_begin quiet
    cvx_precision low
    variable c(num_poles)
    minimize(0.5*(squeeze(data_noise(1,:,d))'-W*c)'*(squeeze(data_noise(1,:,d))'-W*c) + lambda*norm(c*kappa,1))
    
cvx_end

coeffs(:,d) = c;
h.c_cvx = c;
h.p_cvx = cvx_optval;

end
%% calculating the least square error
sqerror  = diag((squeeze(data_noise)-W*coeffs)'*(squeeze(data_noise)-W*coeffs))/T;
plot(sqerror);
 sum(sqerror)/N
figure;
samplenum = 5;
plot(squeeze(data_noise(1,:,samplenum)),'*');
hold on;
y_hat = W*coeffs;
plot(y_hat(:,samplenum),'o');
hold off;
legend groundtruth estimated
xlabel('t');
ylabel('input and output signal');
title('Atomic norm approximation');
