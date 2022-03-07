% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_DProbit"

function [ytp1_pdf, ytp1_pdf_vec] = Pred_DProbit(draws, xtp1, ytp1)
% Inputs:
%   draws: a structure of posterior draws (from "Est_Probit")
%   xtp1: a nx-by-1 vector of x(t+1)
%   ytp1: a scalar of y(t+1)
% Outputs:
%   ytp1_pdf: a scalar of the predictive likelihood
%   ytp1_pdf_vec: a ndraws-by-1 vector of the conditional predictive likelihoods
%   ind_valid: a ndraws-by-1 indicator if each conditional pred likelihood is valid or not

[ndraws,L] = size(draws.phi);
n = size(draws.z,2);

ytp1_pdf_vec = zeros(ndraws,2);
% ind_valid = zeros(ndraws,2);
% count = zeros(2,1);
for drawi = 1:ndraws
    beta = draws.beta(drawi,:)';
%     beta_sparse = [beta(1); draws.beta_nonconst_sparse(drawi,:)'];
    phi = draws.phi(drawi,:)';
%     phi_sparse = draws.phi_sparse(drawi,:)'; 
    ztp1 = flipud(draws.z(drawi,n-L+1:n)');
    
    if ytp1 == 1
        ytp1_pdf_vec(drawi,1) = normcdf(xtp1' * beta + ztp1' * phi);
%         ytp1_pdf_vec(drawi,2) = normcdf(xtp1' * beta_sparse + ztp1' * phi_sparse);
    else
        ytp1_pdf_vec(drawi,1) = 1 - normcdf(xtp1' * beta + ztp1' * phi);
%         ytp1_pdf_vec(drawi,2) = 1 - normcdf(xtp1' * beta_sparse + ztp1' * phi_sparse);
    end
end
ytp1_pdf = mean(ytp1_pdf_vec)';


