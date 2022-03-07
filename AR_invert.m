% let yt = phi1 * ytm1 + ... + phiL * ytmL + ut, t = 1,...,n, y0, y-1, ..., y1mL are given.
% write in matrix form H * y = u + a1 * y0 + ... aL * y1mL
% This function is to compute the matrix H.

function H = AR_invert(phi,n)
% Inputs:
%   phi: a L-by-1 vector of the AR coefficients (phi1, phi2, ..., phiL)
%   n: a scalar of the size of the matrix H
% Outputs:
%   H: a n-by-n matrix


L = length(phi);
H = eye(n);
for j = 1:(n-1)
    if j <= (n-L)
        H(j+1:j+L,j) = -phi;
    else
        H(j+1:n,j) = -phi(1:n-j);
    end
end