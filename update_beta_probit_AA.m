% Consider the model:
% p(beta) ~ N(a, B) where B is diagonal (i.e. independent)
% yt = I{xt' * beta + ut > 0}, t = 1, ..., n
% 
% Given y, x, u, beta, a, B, 
% need to update beta via Gibbs 

function beta = update_beta_probit_AA(y, x, u, beta, beta_prior_mean, beta_prior_var)
% Inputs:
%   y: a n-by-1 vector of the binary target;
%   x: a n-by-K matrix of the regressors;
%   u: a n-by-1 vector of the disturbance;
%   beta: a K-by-1 vector of the regression coef;
%   beta_prior_mean: a K-by-1 vector of the prior mean of beta;
%   beta_prior_var: a K-by-1 vector of the prior variance of beta;
% Outputs:
%   beta: a K-by-1 vector of updated beta.

K = length(beta);
for j = 1:K
    xaa = x(:,j);
    zaa = u + x * beta - xaa * beta(j);
    [lb, ub] = probit_AA_bounds(y, xaa, zaa);
    
    bmeanj = beta_prior_mean(j);
    bstdj = sqrt(beta_prior_var(j)); 
    beta(j) = bmeanj + bstdj * trandn((lb-bmeanj)/bstdj, (ub-bmeanj)/bstdj); 
end


