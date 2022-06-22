% Regularized horseshoe prior
% betaj ~ N(0, c * tau * lambdaj / (c + tau * lambdaj)), j = 1, ..., K
% 
% sqrt(lambdaj) ~ C+(0,1);
% sqrt(tau) ~ C+(0,taud);
% c ~ IG(a,b)
%
% yt = u + xt' * beta + ut, ut is standard Gaussian
% integrate out beta when sampling c, tau lambda
%
% beta = [beta1; beta2] where each subset has a different RHS
% special for use in the AA parameterization of TVP model
%
% note that the intercept u ~ N(0,su) is not regularized

function [logtau1, loglambda1, logc1, logtau2, loglambda2, logc2, count_ctl,...
    ctl_mean, ctl_cov, logrw_ctl, drawi_start_ctl, ...
    logrw_start_ctl] = regularized_HS_int3(xtimesy, xtimesx, logtau1, loglambda1, logc1,...
    logtau2, loglambda2, logc2, taud1, taud2, a, b, drawi, burnin, ctl_mean, ctl_cov,...
    logrw_ctl, drawi_start_ctl, logrw_start_ctl, AMH_c, pstar, su)
% Inputs:
%  y: a n-by-1 vector of targets;
%  x: a n-by-K matrix of regressors;
%  xtimesx: a K-by-K matrix of x' * x;
%  logtau: a scalar of log(tau);
%  loglambda: a K-by-1 vector of log(lambda);
%  logc: a scalar of log(c);
%  taud: a scalar of the scaling factor for tau;
%  a: a scalar of the location hyperparameter for c;
%  b: a scalar of the scale hyperparameter for c;
%  drawi: a scalar of the index;
%  burnin: a scalar of the number of burn-ins;
%  ctl_mean: a (K+2)-by-1 vector of the posterior mean stacking c, tau, lambda;
%  ctl_cov: a (K+2)-by-(K+2) matrix of the posterior covariance;
%  logrw_ctl: a scalar of the log stdev of the RW proposal;
%  drawi_start_ctl: a scalar for tuning the RW proposal;
%  logrw_start_ctl: a scalar;
%  AMH_c: a scalar;
%  pstar: a scalar of target acceptance rate
% Outputs:
%  logtau: a scalar of log(tau);
%  loglambda: a K-by-1 vector of log(lambda);
%  logc: a scalar of log(c);
%  count_ctl: an indicator of MH acceptance;
%  ctl_mean: a (K+2)-by-1 vector of the posterior mean stacking c, tau, lambda;
%  ctl_cov: a (K+2)-by-(K+2) matrix of the posterior covariance;
%  logrw_ctl: a scalar of the log stdev of the RW proposal;
%  drawi_start_ctl: a scalar for tuning the RW proposal;
%  logrw_start_ctl: a scalar;


%% Preparation
d2 = taud1*taud1;
dd2 = taud2*taud2;
K1 = length(loglambda1);
K2 = length(loglambda2);
KK = K1 + K2 + 4;
ctl = [logc1; logtau1; loglambda1; logc2; logtau2; loglambda2];

ctl_old = ctl;
logc1_old = logc1; 
logtau1_old = logtau1; 
loglambda1_old = loglambda1;
logc2_old = logc2; 
logtau2_old = logtau2; 
loglambda2_old = loglambda2;
count_ctl = 0;


%% Propose new para
if drawi < 100
    A = eye(KK);
else  
    A = ctl_cov + 1e-6 * eye(KK) / drawi; %add a small constant
end
eps = mvnrnd(zeros(KK,1),A)'; %correlated normal
ctl_new = ctl_old + exp(logrw_ctl) * eps;
logc1_new = ctl_new(1);
logtau1_new = ctl_new(2);
loglambda1_new = ctl_new(3:K1+2);
logc2_new = ctl_new(K1+3);
logtau2_new = ctl_new(K1+4);
loglambda2_new = ctl_new(K1+5:KK);


%% MH step
logprior1 = -a*logc1_new - b/exp(logc1_new) + a*logc1_old + b/exp(logc1_old)...
            -a*logc2_new - b/exp(logc2_new) + a*logc2_old + b/exp(logc2_old); %c
logprior2 = -0.5*logtau1_new - log(1+d2*exp(-logtau1_new)) ...
    + 0.5*logtau1_old + log(1+d2*exp(-logtau1_old))...
            -0.5*logtau2_new - log(1+dd2*exp(-logtau2_new)) ...
    + 0.5*logtau2_old + log(1+dd2*exp(-logtau2_old)); %tau
logprior3 = -0.5*sum(loglambda1_new) - sum(log(1+exp(-loglambda1_new))) ...
    + 0.5*sum(loglambda1_old) + sum(log(1+exp(-loglambda1_old)))...
            -0.5*sum(loglambda2_new) - sum(log(1+exp(-loglambda2_new))) ...
    + 0.5*sum(loglambda2_old) + sum(log(1+exp(-loglambda2_old))); %lambda
logprior = logprior1 + logprior2 + logprior3;

siginv = [su; exp(-logc1_old) + exp(-logtau1_old -loglambda1_old); ...
    exp(-logc2_old) + exp(-logtau2_old -loglambda2_old)];
Binv = diag(siginv) + xtimesx;
L = chol(Binv)';
logdet_Binv = 2 * sum(log(diag(L)));
tmp = L\xtimesy;
loglike_old = 0.5*sum(log(siginv)) - 0.5*logdet_Binv + 0.5*(tmp'*tmp);

siginv = [su; exp(-logc1_new) + exp(-logtau1_new -loglambda1_new); ...
    exp(-logc2_new) + exp(-logtau2_new -loglambda2_new)];
Binv = diag(siginv) + xtimesx;
L = chol(Binv)';
logdet_Binv = 2 * sum(log(diag(L)));
tmp = L\xtimesy;
loglike_new = 0.5*sum(log(siginv)) - 0.5*logdet_Binv + 0.5*(tmp'*tmp);

logprob = logprior + loglike_new - loglike_old;
if log(rand) <= logprob
    ctl = ctl_new;
    if drawi > burnin
        count_ctl = 1;
    end
end


p = exp(min(0,logprob));
ei = max(200, drawi/KK);
ei_start = max(200, drawi_start_ctl/KK);
dd = max(ei - ei_start, 20);
logrwj = logrw_ctl + AMH_c * (p - pstar)/dd;
if abs(logrwj - logrw_start_ctl) > 1.0986 %log(3) ~= 1.0986 
    drawi_start_ctl = drawi;
    logrw_start_ctl = logrw_ctl;
end %restart when useful to allow for larger movement    
logrw_ctl = logrwj; %update proposal stdev


ctl_mean_old = ctl_mean;
ctl_cov_old = ctl_cov;
ctl_mean = (ctl_mean_old * (drawi-1) + ctl) / drawi;
ctl_cov = (drawi - 1) * (ctl_cov_old + ctl_mean_old * ctl_mean_old') / drawi + ...
    ctl * ctl' / drawi - ctl_mean * ctl_mean'; %update the sample covariance 


logc1 = ctl(1);
logtau1 = ctl(2);
loglambda1 = ctl(3:K1+2);
logc2 = ctl(K1+3);
logtau2 = ctl(K1+4);
loglambda2 = ctl(K1+5:KK);


