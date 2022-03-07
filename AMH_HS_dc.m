% Consider the horseshoe prior: bj ~ N(0, tau * tauj), sqrt(tau) is double cauchy , 
% tauj ~ IB(0.5,0.5),
% Draw tau conditional on bj, tauj via adaptive MH
% logp(logtau|) = 0.5*logtau + log(logtau/(tau-1))

function [tau, count_tau, logrw_tau, drawi_start_tau, logrw_start_tau] = AMH_HS_dc(b2, tau,...
    tauj, pstar, logrw_tau, drawi_start_tau, logrw_start_tau, ...
    drawi, burnin)
% function [tau, count_tau, logrw_tau, drawi_start_tau, logrw_start_tau, ...
%     logtau_mean, logtau_var] = AMH_HS_tau(b2, tau,...
%     tauj, pstar, logrw_tau, drawi_start_tau, logrw_start_tau, logtau_mean, logtau_var,...
%     drawi, burnin, tau_a)
% Inputs:
%   b2: a K-by-1 vector of squared coefficients;
%   tau: a scalar of old tau;
%   tauj: a K-by-1 vector of local variances;
%   pstar: a scalar of target acceptance rate (0.44 for univariate case)
%   logrw_tau: a scalar of old RW scalar;
%   drawi_start_tau: a scalar of starting draw; (adjust poor starting value of rw_tau)
%   logrw_start_tau: a scalar of starting rw_tau;
%   drawi: a scalar indexing the MCMC draw;
%   burnin: a scalar of burn-in length;
% Outputs:
%   tau: a scalar of updated tau;
%   count_tau: a 0/1 indicator of MH acceptance;
%   logrw_tau: a scalar of updated RW scalar;
%   drawi_start_tau: a scalar of updated starting draw; (adjust poor starting value of rw_tau)
%   logrw_start_tau: a scalar of updated starting rw_tau;

K = length(b2);
count_tau = 0;

% if drawi < 100
%     A = 1;
% else  
%     A = sqrt(logtau_var); 
% end
A = 1;

tau_old = tau;
logtau_old = log(tau_old);
logtau_new = logtau_old + exp(logrw_tau) * A * randn;
tau_new = exp(logtau_new);

logprior_old = log(logtau_old / (tau_old - 1)) + 0.5 * logtau_old;
logprior_new = log(logtau_new / (tau_new - 1)) + 0.5 * logtau_new; %p(log(tau))

tmp = 0.5 * sum(b2./tauj);
loglike_old = -0.5*K*logtau_old - tmp / tau_old;
loglike_new = -0.5*K*logtau_new - tmp / tau_new;

logprob = logprior_new + loglike_new - logprior_old - loglike_old;
if log(rand) <= logprob
    tau = tau_new;
    if drawi > burnin
        count_tau = 1;
    end
end


p = exp(min(0,logprob));
% ei = max(200, drawi/K);
% ei_start = max(200, drawi_start_v/K);
d = max(drawi - drawi_start_tau, 20);
logrwj = logrw_tau + (p - pstar)/(d * pstar * (1-pstar));
if abs(logrwj - logrw_start_tau) > 1.0986 %log(3) ~= 1.0986 
    drawi_start_tau = drawi;
    logrw_start_tau = logrw_tau;
end %restart when useful to allow for larger movement    
logrw_tau = logrwj; %update proposal stdev 


% logtau = log(tau);
% logtau_mean_old = logtau_mean;
% logtau_var_old = logtau_var;
% logtau_mean = (logtau_mean_old * (drawi-1) + logtau) / drawi;
% logtau_var = (drawi - 1) * (logtau_var_old + logtau_mean_old * logtau_mean_old') / drawi + ...
%     logtau * logtau / drawi - logtau_mean * logtau_mean'; %update the sample covariance



