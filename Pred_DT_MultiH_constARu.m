% Compute predictive likelihood p(y(t+1) | y(1:t), x(1:t+1))
% Use after running "Est_Probit_TVP"

function [ytph_logpdf, ytph_logpdf_vec] = Pred_DT_MultiH_constARu(draws, xmat, ytph, h, ind_sim)
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

ytph_logpdf_vec = zeros(ndraws,1);
for drawi = 1:ndraws
    zt = draws.z(drawi,n);
    bt_mean = draws.bn_mean(drawi,:)';
    bt_cov = draws.bn_cov{drawi};
    v2 = (draws.v(drawi,:)).^2;
    u = draws.alpha0(drawi,1);
    phi = draws.alpha0(drawi,K+1);
    ytph_logpdf_drawi = dt_multi_integ_constARu(zt, xmat, u, phi, bt_mean, bt_cov, diag(v2), h, ytph);

    ytph_logpdf_vec(drawi) = ytph_logpdf_drawi;
%     if ytph == 1
%         ytph_pdf_vec(drawi,1) = ytph_pdf_drawi;
%     else
%         ytph_pdf_vec(drawi,1) = 1 - ytph_pdf_drawi;
%     end
end
logpdf_mean = mean(ytph_logpdf_vec);
ytph_logpdf = logpdf_mean + log(mean(exp(ytph_logpdf_vec - logpdf_mean)));


