% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP"

function [ytph_logpdf, ytph_logpdf_vec] = Pred_PT_MultiH_constu(draws, xtph, ytph, h)
% Inputs:
%   draws: a structure of posterior draws (from "Est_Probit_TVP")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

[ndraws,n] = size(draws.z);
K = length(xtph);

ytph_logpdf_vec = zeros(ndraws,1);
for drawi = 1:ndraws
    beta = [draws.alpha0(drawi,1); draws.bn_mean(drawi,:)'];
    v2 = (draws.v(drawi,:)).^2;
    beta_cov = h * diag(v2) + draws.bn_cov{drawi};
    beta_std = sqrt(1 + xtph(2:K)' * beta_cov * xtph(2:K));
    
    tmp = xtph' * beta / beta_std;
    if ytph == 0
        tmp = -tmp; %reverse sign 
    end    
    if tmp > -38
        lognormcdf = log(normcdf(tmp)); %numerically feasible
    else
        lognormcdf = -0.5*log(2*pi) - 0.5*tmp*tmp - log(-tmp); %approximation
    end
    ytph_logpdf_vec(drawi) = lognormcdf;    
    
%     if ytph == 1
%         ytph_pdf_vec(drawi,1) = normcdf(xtph' * beta / beta_std);
%     else
%         ytph_pdf_vec(drawi,1) = 1 - normcdf(xtph' * beta / beta_std);
%     end
end
% ytph_pdf = mean(ytph_pdf_vec)';
logpdf_mean = mean(ytph_logpdf_vec);
ytph_logpdf = logpdf_mean + log(mean(exp(ytph_logpdf_vec - logpdf_mean)));





