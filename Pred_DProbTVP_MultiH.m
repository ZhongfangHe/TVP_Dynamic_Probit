% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP"

function [ytph_pdf, ytph_pdf_vec] = Pred_DProbTVP_MultiH(draws, xmat, ytph, h, ind_sim)
% Inputs:
%   draws: a structure of posterior draws (from "Est_Probit_TVP")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

[ndraws,n] = size(draws.z);
K = size(xmat,2);

ytph_pdf_vec = zeros(ndraws,2);
% ind_valid = zeros(ndraws,2);
% count = zeros(2,1);
% beta = zeros(K+1,1);
% beta_sparse = zeros(K+1,1);
for drawi = 1:ndraws
%     for j = 1:K+1
%         beta(j) = draws.beta{j}(drawi,n);
%         beta_sparse(j) = draws.beta_sparse{j}(drawi,n);
%     end
    
%     xztp1 = [xmat; zt];
    
%     v2 = (draws.v(drawi,:)).^2;
%     beta_std = sqrt(1 + sum((xztp1.^2).*v2'));
%     v2_sparse = (draws.v_sparse(drawi,:)).^2;
%     beta_sparse_std = sqrt(1 + sum((xztp1.^2).*v2_sparse'));
    
    
%     ytph_pdf_drawi = dprob_tvp_multi_horizon(zt, xmat, beta, diag(v2), h, ind_sim);
    zt = draws.z(drawi,n);
    bt_mean = draws.bn_mean(drawi,:)';
    bt_cov = draws.bn_cov{drawi};
    v2 = (draws.v(drawi,:)).^2;
    ytph_pdf_drawi = dprob_tvp_multi_integ(zt, xmat, bt_mean, bt_cov, diag(v2), h, ind_sim);
%     ytph_pdf_sparse_drawi = ytph_pdf_drawi;
    if ytph == 1
        ytph_pdf_vec(drawi,1) = ytph_pdf_drawi;
%         ytph_pdf_vec(drawi,2) = ytph_pdf_sparse_drawi;
    else
        ytph_pdf_vec(drawi,1) = 1 - ytph_pdf_drawi;
%         ytph_pdf_vec(drawi,2) = 1 - ytph_pdf_sparse_drawi;
    end
end
ytph_pdf = mean(ytph_pdf_vec)';


