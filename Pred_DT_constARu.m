% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP_constARu"

function [ytp1_logpdf, ytp1_logpdf_vec] = Pred_DT_constARu(draws, xtp1, ytp1)
% Inputs:
%   draws: a structure of posterior draws (from "Est_Probit_TVP")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

[ndraws,n] = size(draws.z);
K = length(xtp1);

ytp1_logpdf_vec = zeros(ndraws,1);
for drawi = 1:ndraws
    alpha = [draws.alpha0(drawi,1); draws.bn_mean(drawi,:)'; draws.alpha0(drawi,K+1)];
    
    v2 = (draws.v(drawi,:)).^2;
    beta_cov = diag(v2) + draws.bn_cov{drawi};     
%     beta_std = sqrt(1 + xztp1' * beta_cov * xztp1);
    beta_std = sqrt(1 + xtp1(2:K)' * beta_cov * xtp1(2:K));

    zt = draws.z(drawi,n);
    xztp1 = [xtp1; zt];
    tmp = xztp1' * alpha / beta_std;
    if ytp1 == 0
        tmp = -tmp; %reverse sign 
    end    
    if tmp > -38
        lognormcdf = log(normcdf(tmp)); %numerically feasible
    else
        lognormcdf = -0.5*log(2*pi) - 0.5*tmp*tmp - log(-tmp); %approximation
    end
    ytp1_logpdf_vec(drawi) = lognormcdf;
   
    
%     if ytp1 == 1
%         ytp1_pdf_vec(drawi,1) = normcdf(xztp1' * alpha / beta_std);
%     else
%         ytp1_pdf_vec(drawi,1) = 1 - normcdf(xztp1' * alpha / beta_std);
%     end
end
logpdf_mean = mean(ytp1_logpdf_vec);
ytp1_logpdf = logpdf_mean + log(mean(exp(ytp1_logpdf_vec - logpdf_mean)));


