% Consider the linear regression model with spike-slab prior:
% yt = xt' * b + N(0, sig2t), where sig2t is given
% bk ~ N(0, indk * sk + (1-indk) * c * sk), k=1,...,K, t=1,...,n.
% p(indk = 1) = q, p(sk) = IG(s_a0, s_b0), c~=0 (e.g. 1e-6)
%
% Draw ind conditional on beta
%
% Update prior parameters given beta

function [s, q] = SpikeSlab_PriorUpdate_sq_marg(beta, ind, c, s_prior)
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
K = length(beta);
beta2 = beta.^2;
c_inv = 1/c;
chalf_inv = 1/sqrt(c);


%% Update indicator probability
sum_ind = sum(ind);
q_a = 1 + sum_ind;
q_b = 1+ K - sum_ind;
q = betarnd(q_a, q_b);


%% Update slab variance
s_a = s_prior(1) + 0.5;
s_b = s_prior(2) + 0.5 * beta2 .* (ind + (1-ind)*c_inv);
s = 1./gamrnd(s_a, 1./s_b);


% %% Update indicator (conditional on beta)
% tmp = exp(-0.5 * (c_inv - 1) * beta2./s);
% ind1_prob = q ./ (q + (1 - q) * chalf_inv * tmp);
% ind = double(rand(K,1) <= ind1_prob);


% %% Update beta
% psi = ind.*s + c * (1-ind).*s;
% sig = sqrt(sig2);
% xstar = x ./ repmat(sig,1,K);
% ystar = y ./ sig;
% A_inv = diag(1./psi) + xstar' * xstar;
% A_inv_half = chol(A_inv);
% a = A_inv \ (xstar' * ystar);
% beta = a + A_inv_half \ randn(K,1);



