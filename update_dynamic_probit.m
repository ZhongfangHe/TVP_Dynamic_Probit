% Update of dynamic probit model:
% yt = I{zt > 0} or sign(zt),
% zt = xt' * beta + phi1 * ztm1 + ... + phiL * ztmL + N(0,1),
% 
% prior: beta ~ N(a0,b0), phi ~ N(c0,d0)
% alpha = [beta; phi]

function [z, alpha, zi] = update_dynamic_probit(y, x, z, alpha, zi, ...
    alpha_prior_mean, alpha_prior_cov_inv, zi_prior_mean, zi_prior_cov_inv)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   z: a n-by-1 latent index,
%   alpha: a (K+L)-by-1 vector of reg coefs and AR coefs,
%   zi: a L-by-1 vector of AR initials z1mL, z2mL, ..., z0.
%   alpha_prior_mean: a (K+L)-by-1 vector of prior mean for alpha,
%   alpha_prior_cov_inv: a (K+L)-by-(K+L) matrix of prior precision,
%   zi_prior_mean: a L-by-1 vector of prior mean for zi,
%   zi_prior_cov_inv: a L-by-L matrix of prior precision,

% Outputs:
%   z: updated z,
%   alpha: updated alpha,
%   zi: updated zi.


[n,K] = size(x);
L = length(zi);
beta = alpha(1:K);
phi = alpha(K+1:K+L);


% Draw latent index z
% xb = x * beta; 
% zfull = [zi; z];
% for t = 1:n
%     zt_prior_mean = xb(t) + flipud(zfull(t:t+L-1))' * phi;
%     zt_prior_var = 1;
%     zt_prior_var_inv = 1/zt_prior_var;
% 
%     if t < n %prior + likelihood
%         m = min(n-t, L);
%         zt_x = phi(1:m);
%         zt_y1 = z(t+1:t+m) - xb(t+1:t+m);
%         zt_y2 = AR_X_mat(zfull(t-L+1+L:t+m-1+L), L) * phi - phi(1:m) * z(t);
%         zt_y = zt_y1 - zt_y2; 
% 
%         zt_post_var_inv = zt_prior_var_inv + zt_x' * zt_x;
%         zt_post_var = 1/zt_post_var_inv;
%         zt_post_mean = zt_post_var * (zt_prior_var_inv * zt_prior_mean + zt_x' * zt_y);
%         zt_post_std = sqrt(zt_post_var);
%     else %t=n, no likelihood, only prior
%         zt_post_mean = zt_prior_mean;
%         zt_post_std = sqrt(zt_prior_var);
%     end
% 
%     tmp = -zt_post_mean/zt_post_std; 
%     if y(t) > 0 %zt > 0
%         zt = zt_post_mean + zt_post_std * trandn(tmp, Inf);
%     else %y(t) = 0 <=> zt <= 0
%         zt = zt_post_mean + zt_post_std * trandn(-Inf,tmp);
%     end
% 
%     z(t) = zt;
%     zfull(t+L) = zt;
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Draw z (HMC)
% [H, mu] = TVP_AR_invert(0.8*ones(n,1));
[H, mu] = TVP_AR_invert(phi*ones(n,1));
F = diag(2 * y - 1);
g = zeros(n,1);
M = H' * H;
mu_r = H' * (mu * zi + x*beta);
z_old = z;
z_new = HMC_exact(F, g, M, mu_r, false, 2, z_old);
z = z_new(:,2);
% %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Draw initial latent index
zi_x = AR_slope_matrix(phi);
if L > 1
    tmp = convolute(phi(1:L-1), z(1:L-1));
    zi_y = z(1:L) - x(1:L,:) * beta - [0; tmp]; 
else
    zi_y = z(1) - x(1,:) * beta; %only for L = 1
end
zi_b_inv = zi_prior_cov_inv + zi_x' * zi_x;
zi_b = zi_b_inv\eye(L);
if ~any(zi_prior_mean) 
    zi_a = zi_b * (zi_x' * zi_y);
else
    zi_a = zi_b * (zi_prior_cov_inv * zi_prior_mean + zi_x' * zi_y);
end
zi = mvnrnd(zi_a, zi_b)';  




% Draw reg coefs beta and AR coefs phi   
alpha_x = [x  AR_X_mat([zi; z(1:n-1)], L)];
alpha_y = z;   
b_inv = alpha_prior_cov_inv + alpha_x' * alpha_x;
b = b_inv\eye(L+K);
if ~any(alpha_prior_mean) 
    a = b * (alpha_x' * alpha_y);
else
    a = b * (alpha_prior_cov_inv * alpha_prior_mean + alpha_x' * alpha_y);
end
alpha = mvnrnd(a, b)';


  
    






