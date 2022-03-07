% Consider the TVP-AR model: zt = phit * ztm1
% Write in matrix form: H * z = mu * z0

function [H, mu] = TVP_AR_invert(phi_vec)
% Inputs:
%   phi_vec: a n-by-1 vector of TVP AR(1) coefficients
% Outputs:
%   H: a n-by-n matrix of a lower-bidiagonal matrix with phi2 to phin
%   mu: a n-by-1 vector of zeros with the 1st element being phi1

n = length(phi_vec);

mu = zeros(n,1);
mu(1) = phi_vec(1);

H = eye(n);
for t = 2:n
    H(t,t-1) = -phi_vec(t);
end

