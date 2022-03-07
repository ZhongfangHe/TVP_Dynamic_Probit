% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP"

function [ytph_pdf, ytph_pdf_vec] = Pred_ProbTVP_MultiH(draws, xtph, ytph, h)
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

ytph_pdf_vec = zeros(ndraws,2);
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
%     beta_std = sqrt(1 + h * sum((xtph.^2).*v2'));
%     v2_sparse = (draws.v_sparse(drawi,:)).^2;
%     beta_sparse_std = sqrt(1 + h * sum((xtph.^2).*v2_sparse'));

    beta = draws.bn_mean(drawi,:)';
    v2 = (draws.v(drawi,:)).^2;
    beta_cov = h * diag(v2) + draws.bn_cov{drawi};
    beta_std = sqrt(1 + xtph' * beta_cov * xtph);
    
%     beta_sparse = beta;
%     beta_sparse_std = beta_std;
    
    if ytph == 1
        ytph_pdf_vec(drawi,1) = normcdf(xtph' * beta / beta_std);
%         ytph_pdf_vec(drawi,2) = normcdf(xtph' * beta_sparse / beta_sparse_std);
    else
        ytph_pdf_vec(drawi,1) = 1 - normcdf(xtph' * beta / beta_std);
%         ytph_pdf_vec(drawi,2) = 1 - normcdf(xtph' * beta_sparse / beta_sparse_std);
    end
end
ytph_pdf = mean(ytph_pdf_vec)';


