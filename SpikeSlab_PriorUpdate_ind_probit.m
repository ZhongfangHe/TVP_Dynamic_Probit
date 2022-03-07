% Consider the linear regression model with spike-slab prior:
% yt = xt' * b + N(0, sig2t), where sig2t is given
% bk ~ N(0, indk * sk + (1-indk) * c * sk), k=1,...,K, t=1,...,n.
% p(indk = 1) = q, p(sk) = IG(s_a0, s_b0), c~=0 (e.g. 1e-6)
%
% Draw ind marginalizing out beta
% Draw ind by single moves
% Need a function "indicator_matrix" from the folder "2021Jul/Functions/Mixture_Innovation"

function [ind_x, ind_z] = SpikeSlab_PriorUpdate_ind_probit(y, x, z, sig2, c, ind_x, ind_z, ...
    s_x, s_z, q_x, q_z)
% Inputs:
%   y: a n-by-1 vector of targets;
%   x: a n-by-K matrix of regressors;
%   sig2: a n-by-1 vector of the noise variance;
%   beta: a K-by-1 vector of the regression coefficients;
%   ind: a K-by-1 vector of mixture indicator;
%   c: a sclar of spike constant (e.g. 1e-6);
%   s_prior: a 2-by-1 vector of the hyper-parameters for s (IG prior);
% Outputs:
%   beta: a K-by-1 vector of the updated regression coefficients;
%   ind: a K-by-1 vector of the updated mixture indicator;
%   s: a K-by-1 vector of slab variances;
%   q: a scalar of indicator probability (ind = 1);



% [n,K] = size(x);
Kx = size(x,2);
Kz = size(z,2);
K = Kx + Kz;
% beta2 = beta.^2;
sig = sqrt(sig2);
xstar = [x z]./ repmat(sig,1,K);
ystar = y ./ sig;
xxstar = xstar' * xstar;
xystar = xstar' * ystar;

% c_inv = 1/c;
chalf_inv = 1/sqrt(c);


% %% Update indicator probability
% sum_ind = sum(ind);
% q_a = 1 + sum_ind;
% q_b = 1+ K - sum_ind;
% q = betarnd(q_a, q_b);
% 
% 
% %% Update slab variance
% s_a = s_prior(1) + 0.5;
% s_b = s_prior(2) + 0.5 * beta2 .* (ind + (1-ind)*c_inv);
% s = 1./gamrnd(s_a, 1./s_b);


%% Update indicator (marginalize out beta, single move)
ind = [ind_x;  ind_z];
s = [s_x; s_z];
for j = 1:K %iterate over possible scenarios
    indj = ind;
    
    indj(j) = 1;
    psi = indj.*s + c * (1-indj).*s;
    A_inv = diag(1./psi) + xxstar;
    A_inv_half = chol(A_inv);
    a = A_inv \ xystar;
    logdet_A_inv = 2 * sum(log(diag(A_inv_half)));
    indj1_log_like = - 0.5*logdet_A_inv + 0.5*a'*xystar;
    
    indj(j) = 0;
    psi = indj.*s + c * (1-indj).*s;
    A_inv = diag(1./psi) + xxstar;
    A_inv_half = chol(A_inv);
    a = A_inv \ xystar;
    logdet_A_inv = 2 * sum(log(diag(A_inv_half)));
    indj0_log_like = - 0.5*logdet_A_inv + 0.5*a'*xystar;    
    
    if j <= Kx
        q = q_x;
    else
        q = q_z;
    end
    indj_prob = q / (q + (1-q)*chalf_inv*exp(indj0_log_like - indj1_log_like));
    ind(j) = double(rand <= indj_prob);   
end
ind_x = ind(1:Kx);
ind_z = ind(Kx+1:K);



% %% Update beta
% psi = ind.*s + c * (1-ind).*s;
% A_inv = diag(1./psi) + xxstar;
% A_inv_half = chol(A_inv);
% a = A_inv \ xystar;
% beta = a + A_inv_half \ randn(K,1);



