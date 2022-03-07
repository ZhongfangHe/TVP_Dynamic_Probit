% Consider a dynamic probit model: zt = c * ztm1 + a' * xt + ut, ut ~ N(0,1)
% Iterate h-steps ahead: ztph ~ N(zxt'*d, e)

function ytph_pdf = dprob_multi_horizon(zt, xmat, beta, h)
% Inputs:
%   zt: a scalar of the starting value;
%   xmat: a h-by-K matrix of the regressors for prediction periods;
%   betat: a (K+1)-by-1 vector of the starting coef with the order of at,ct;
%   v: a (K+1)-by-(K+1) matrix of the covariance matrix of betat
%   h: a scalar of the number of horizons
% Outputs:
%   ytph_pdf: a scalar of the predictive density p(ytph=1) = p(ztph > 0)


K = size(xmat,2);
Kh = K * h;

alpha = beta(1:K);
phi = beta(K+1);
phi2 = phi^2;
phih = phi^h;
phi2h = phi2^h;


%% compute the mean and variance of ztph
tmp = phi2.^(1:(h-1));
et = 1 + sum(tmp);
% et = (1 - phi2h)/(1 - phi2);

tmp = phi.^(0:(h-1));
tmp = flipud(tmp');
dt = [phih; kron(tmp,alpha)]; 

zxt = [zt; reshape(xmat',Kh,1)];

ytph_pdf = normcdf((zxt'*dt)/sqrt(et));



