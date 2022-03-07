% Consider a TVP model: zt = ct * ztm1 + at' * xt + ut, ut ~ N(0,1)
% Iterate h-steps ahead: ztph ~ N(zxt'*dt, et)

% Use Kalman filter to integrate out ct and at

function ytph_pdf = dprob_tvp_multi_integ(zt, xmat, bt_mean, bt_cov, v, h, ind_sim)
% Inputs:
%   zt: a scalar of the starting value;
%   xmat: a h-by-K matrix of the regressors for prediction periods;
%   bt_mean: a (K+1)-by-1 vector of the starting coef with the order of at,ct;
%   v: a (K+1)-by-(K+1) matrix of the covariance matrix of betat
%   h: a scalar of the number of horizons
% Outputs:
%   ytph_pdf: a scalar of the predictive density p(ytph=1) = p(ztph > 0)


K = size(xmat,2);
Kh = K * h;


%% mean and cov for ctp1, ctp2, ..., ctph
% simulate or set to equal ct
ct_mean = bt_mean(K+1);
ct_var = bt_cov(K+1,K+1);
ch_mean = ct_mean * ones(h,1); 
A = rw_cov(h);
ch_cov = v(K+1,K+1) * A + ct_var * ones(h,h);
if ind_sim == 1 %simulate
    ch = mvnrnd(ch_mean, ch_cov)';
else %equal to ct
    ch = ch_mean;
end


%% mean and cov for atp1, atp2, ..., atph
at_mean = bt_mean(1:K);
at_cov = bt_cov(1:K,1:K);
ah_mean_marginal = kron(ones(h,1),at_mean);
ah_cov_marginal = kron(A,v(1:K,1:K)) + kron(ones(h,h),at_cov);
cov_ah_ch = kron(ones(h,h), bt_cov(1:K,K+1)); 
ah_mean = ah_mean_marginal + cov_ah_ch * (ch_cov \ (ch - ch_mean)); %conditional mean
ah_cov = ah_cov_marginal - cov_ah_ch * (ch_cov \ cov_ah_ch'); %conditional cov


%% Assemble the mean and var for interated ztph
ch_prod = cumprod(flipud(ch));
c_star = ch_prod(h); 
c_hat = [flipud(ch_prod(1:h-1)); 1];

xmat_hat = xmat .* repmat(c_hat,1,K);
xvec_hat = reshape(xmat_hat',Kh,1);

ztph_mean = c_star * zt + xvec_hat' * ah_mean;
ztph_var = c_hat' * c_hat + xvec_hat' * ah_cov * xvec_hat;


%% Compute pdf
ytph_pdf = normcdf(ztph_mean/sqrt(ztph_var));





