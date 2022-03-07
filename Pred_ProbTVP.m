% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP"

function [ytp1_pdf, ytp1_pdf_vec] = Pred_ProbTVP(draws, xtp1, ytp1)
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

ytp1_pdf_vec = zeros(ndraws,2);
% ind_valid = zeros(ndraws,2);
% count = zeros(2,1);
% beta = zeros(K,1);
% beta_sparse = zeros(K,1);
for drawi = 1:ndraws
%     for j = 1:K
%         beta(j) = draws.beta{j}(drawi,n);
%         beta_sparse(j) = draws.beta_sparse{j}(drawi,n);
%     end
%     
%     v2 = (draws.v(drawi,:)).^2;
%     beta_std = sqrt(1 + sum((xtp1.^2).*v2'));
%     v2_sparse = (draws.v_sparse(drawi,:)).^2;
%     beta_sparse_std = sqrt(1 + sum((xtp1.^2).*v2_sparse'));
    
    beta = draws.bn_mean(drawi,:)';
    v2 = (draws.v(drawi,:)).^2;
    beta_cov = diag(v2) + draws.bn_cov{drawi};
    beta_std = sqrt(1 + xtp1' * beta_cov * xtp1);
    
%     beta_sparse = beta;
%     beta_sparse_std = beta_std;
    
    if ytp1 == 1
        ytp1_pdf_vec(drawi,1) = normcdf(xtp1' * beta / beta_std);
%         ytp1_pdf_vec(drawi,2) = normcdf(xtp1' * beta_sparse / beta_sparse_std);
    else
        ytp1_pdf_vec(drawi,1) = 1 - normcdf(xtp1' * beta / beta_std);
%         ytp1_pdf_vec(drawi,2) = 1 - normcdf(xtp1' * beta_sparse / beta_sparse_std);
    end
end
ytp1_pdf = mean(ytp1_pdf_vec)';


