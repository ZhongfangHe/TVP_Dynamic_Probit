% Consider the model:
% zt = xt * bt + N(0,1);
% bt = btm1 + N(0,v);
% b0 is given;
%
% Use Kalman filter to compute the sequency of E(bt|y^t) and V(bt|y^t)


function [bn_mean, bn_cov] = TVP_beta_filter(z, x, v, b0)
% Inputs:
%   y: a n-by-p matrix of targets;
%   z: regressors -> a n-by-m matrix if p = 1 or a n-by-1 cell of p-by-m matrices if p > 1;
%   h: measurement noise var/covar -> a n-by-1 vector if p = 1 or a n-by-1 cell of p-by-p matrices if p > 1;
%   q: state noise covar -> a (n-1)-by-1 cell of m-by-m matrices ;
%   P1_inv_times_a1: a m-by-1 vector of the covector of the initial state b1;
%   P1_inv: a m-by-m matrix of the precision matrix of the initial state b1;
% Outputs:
%   bn_mean: a m-by-1 vector of E(bn|y);
%   bn_cov: a m-by-m matrix of V(bn|y);

[n,K] = size(x);
b_mean = zeros(n,K);
b_cov = cell(n,1);
for t = 1:n
    b_cov{t} = zeros(K,K);
end
for t = 1:n
    xt = x(t,:)';
    zt = z(t);  
    if t == 1
        dtm1 = zeros(K,K);
        mtm1 = b0;
    else
        dtm1 = b_cov{t-1};
        mtm1 = b_mean(t-1,:)';
    end
    dv = dtm1 + v;
    tmp = eye(K) - dv * (xt * xt') / (1 + xt' * dv * xt); 
    dt = tmp * dv;
    mt = tmp * mtm1 + dt * xt * zt;
    b_cov{t} = dt;
    b_mean(t,:) = mt';    
end
bn_mean = b_mean(n,:)';
bn_cov = b_cov{n};


