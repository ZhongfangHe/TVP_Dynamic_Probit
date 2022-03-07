% Compute predictive likelihood p(y(t+h) | y(1:t), x(1:t+h))
% Use after running "Est_DProbit"

function [ytph_pdf, ytph_pdf_vec] = Pred_DProbit_MultiH(draws, xmat, ytph, h)
% Inputs:
%   draws: a structure of posterior draws (from "Est_Probit")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
%   h: a scalar of the number of horizons
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

[ndraws,L] = size(draws.phi);
n = size(draws.z,2);

ytph_pdf_vec = zeros(ndraws,2);
% ind_valid = zeros(ndraws,2);
% count = zeros(2,1);
for drawi = 1:ndraws
    beta = draws.beta(drawi,:)';
%     beta_sparse = [beta(1); draws.beta_nonconst_sparse(drawi,:)'];
    phi = draws.phi(drawi,:)';
%     phi_sparse = draws.phi_sparse(drawi,:)'; 
    zt = flipud(draws.z(drawi,n-L+1:n)');
    
    ytph_pdf_drawi = dprob_multi_horizon(zt, xmat, [beta;phi], h);
%     ytph_pdf_sparse_drawi = dprob_multi_horizon(zt, xmat, [beta_sparse;phi_sparse], h);
    if ytph == 1
        ytph_pdf_vec(drawi,1) = ytph_pdf_drawi;
%         ytph_pdf_vec(drawi,2) = ytph_pdf_sparse_drawi;
    else
        ytph_pdf_vec(drawi,1) = 1 - ytph_pdf_drawi;
%         ytph_pdf_vec(drawi,2) = 1 - ytph_pdf_sparse_drawi;
    end
end
ytph_pdf = mean(ytph_pdf_vec)';


