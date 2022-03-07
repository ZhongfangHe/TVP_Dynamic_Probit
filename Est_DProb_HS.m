% Estimate a dynamic probit model:
% yt = I{zt > 0} or sign(zt),
% zt = xt' * beta + phi1 * ztm1 + ... + phiL * ztmL + N(0,1),
% 
% Horseshoe prior for non-constant beta and phi.
%
% ASIS for beta


function draws = Est_DProb_HS(y, x, burnin, ndraws, L, ind_sparse)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   burnin: a scalar of the number of burn-ins,
%   ndraws: a scalar of the number of draws after burn-in,
%   L: a scalar of the number of AR lags,
%   ind_sparse: a scalar if sparsification for non-constant coefs is needed
% Outputs:
%   draws: a structure of the draws.

[n,K] = size(x);
KL = K+L;


%% Prior for initial z
zi_a0 = zeros(L,1); 
zi_b0 = 10*eye(L);
zi_b0_inv = zi_b0\eye(L);
zi = zi_a0 + sqrt(diag(zi_b0)) .* randn(L,1); %initial z

%% Horseshoe prior for non-constant beta and phi
var_const = 10; %prior variance for constant coef

de0_aux = 1/gamrnd(0.5,1);
de0 = 1/gamrnd(0.5,de0_aux);
de_aux = 1./gamrnd(0.5,1,KL-1,1);
de = 1./gamrnd(0.5,de_aux); %hyper-parameters for beta, phi

alpha_a0 = zeros(KL,1);
alpha_b0 = diag([var_const; de0*de]);
% alpha = alpha_a0 + sqrt(diag(alpha_b0)) .* randn(KL,1);
alpha = zeros(KL,1);


%% Initialize the latent index z
z = zeros(n,1);
for t = 1:n
    if y(t) > 0
        z(t) = trandn(0,Inf);
    else
        z(t) = trandn(-Inf,0);
    end
end



%% MCMC
draws.zi = zeros(ndraws,L);
draws.phi = zeros(ndraws,L); 
draws.beta = zeros(ndraws,K); 
draws.alpha_aux = zeros(ndraws,2*KL);
draws.z = zeros(ndraws,n);
draws.p1 = zeros(ndraws,n);
if ind_sparse == 1
    draws.beta_nonconst_sparse = zeros(ndraws,K-1);
    draws.phi_sparse = zeros(ndraws,L);
end
draws.corr_eps = zeros(ndraws,1);
ntotal = burnin + ndraws;
tic;
for drawi = 1:ntotal    
    % Draw alpha and z
    alpha_b0_inv = diag(1./[var_const; de0*de]);
    [z, alpha, zi] = update_dynamic_probit(y, x, z, alpha, zi, ...
            alpha_a0, alpha_b0_inv, zi_a0, zi_b0_inv);
    beta = alpha(1:K);
    phi = alpha(K+1:KL);
    if and(phi > 1, drawi < burnin)
        phi = 1; %reset to avoid numerical error during burnin
        alpha(K+1:KL) = 1;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % ASIS: compute eta = H\eps
    tmp = [x  AR_X_mat([zi; z(1:n-1)], L)]; 
    eps = z - tmp * alpha;   
    H = AR_invert(phi,n);
    eta = H\eps;
    
    % ASIS: re-draw zi and beta in the AA representation
    xstar = H\x;
    M = initial_z_mat(phi,n); %matrix for zi
    zistar = H\M; %z = zistar * zi + xstar * beta + eta
    
    beta_prior_std = sqrt([var_const; de0*de(1:(K-1))]);
    zistar_times_zi_plus_eta = zistar * zi + eta;
    for j = 1:K
        xaa = xstar(:,j);
        zaa = zistar_times_zi_plus_eta + xstar * beta - xaa * beta(j);
        [lb, ub] = probit_AA_bounds(y, xaa, zaa);
        bstdj = beta_prior_std(j); 
        beta(j) = bstdj * trandn(lb/bstdj, ub/bstdj); 
    end
    
    zi_prior_std = sqrt(diag(zi_b0));
    xstar_times_beta_plus_eta = xstar * beta + eta;
    for j = 1:L
        xaa = zistar(:,j);
        zaa = zistar * zi + xstar_times_beta_plus_eta - xaa * zi(j);
        [lb, ub] = probit_AA_bounds(y, xaa, zaa);
        bstdj = zi_prior_std(j); 
        zi(j) = bstdj * trandn(lb/bstdj, ub/bstdj); 
    end 
    
    % ASIS: compute back z
    z = xstar_times_beta_plus_eta + zistar * zi; 
    alpha = [beta;phi];       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % Draw hyper-parameters of alpha
    alpha2 = alpha(2:KL).^2;
    [de0, de0_aux, de, de_aux] = Horseshoe_update_vector(alpha2, de0, de0_aux, de, de_aux);
       
    
    % Compute probability 
    tmp = [x  AR_X_mat([zi; z(1:n-1)], L)]; 
    p1 = normcdf(tmp * alpha);
    
    
    % Sparsify if needed
    if ind_sparse == 1
        alpha_x = [x  AR_X_mat([zi; z(1:n-1)], L)];
        alpha_nonconst_sparse = SAVS_vector(alpha(2:KL),alpha_x(:,2:KL)); %exclude constant
        beta_nonconst_sparse = alpha_nonconst_sparse(1:K-1);
        phi_sparse = alpha_nonconst_sparse(K:KL-1);
    end 
    

    % Compute the first-order autocorrelation of latent index resid
    eps = z - tmp * alpha;
    corr_eps = corr(eps(1:n-1),eps(2:n)); 
    
    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        draws.zi(i,:) = zi';
        draws.phi(i,:) = phi'; 
        draws.beta(i,:) = beta'; 
        draws.alpha_aux(i,:) = [de0  de'  de0_aux  de_aux'];
        draws.z(i,:) = z';
        draws.p1(i,:) = p1';
        if ind_sparse == 1
            draws.beta_nonconst_sparse(i,:) = beta_nonconst_sparse';
            draws.phi_sparse(i,:) = phi_sparse';
        end 
        draws.corr_eps(i) = corr_eps;
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end
end




