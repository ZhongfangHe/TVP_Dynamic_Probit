% Compute the matrix for initial z
% given phi = [phi1  phi2  ...  phiL], compute a matrix:
% [phi1   phi2  phi3  ...  phiL;
% [phi2   phi3  phi4  ...  0;
% [phi3   phi4  phi5  ...  0;
%  ...;
% [phiL   0     0     ...  0;
% [0      0     0     ...  0].

function M = initial_z_mat(phi,n)
% Inputs:
%   phi: a L-by-1 vector fo AR coefficients
%   n: a scalar of the data length
% Outputs:
%   M: a n-by-L matrix of the AR coefficients

L = length(phi);
M = zeros(n,L);
for j = 1:L
    M(j,1:(L-j+1)) = phi(j:L)';
end

