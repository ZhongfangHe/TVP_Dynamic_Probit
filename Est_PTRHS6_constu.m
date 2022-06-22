% Estimate a probit model with TVP and RHS shrinkage
%
% const intercept for the latent variable
%
% Gibbs blocks: beta_star, (z, u,beta0,v,theta0)


function draws = Est_PTRHS6_constu(y, x, burnin, ndraws, ind_sparse, ind_pred)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   burnin: a scalar of the number of burn-ins,
%   ndraws: a scalar of the number of draws after burn-in.
%   ind_sparse: an indicator if sparsification for non-constant coefs is needed (not updated, should be 0)
%   ind_pred: an indicator if prediction is needed (run a Kalman filter)
% Outputs:
%   draws: a structure of the draws.

[n,K] = size(x);
minNum = 1e-100;
maxNum = 1e100;


%% Priors: 
% intercept, u ~ N(0,su)
su = 10;
u = sqrt(su) * randn;


% initial beta, beta0 ~ N(0, psil)
taul0 = 1/n;
taul_d = 1/gamrnd(0.5,taul0^2);
taul = 1/gamrnd(0.5, taul_d); %global variance
logtaul = log(taul);

taujl_d = 1./gamrnd(0.5,1,K-1,1);
taujl = 1./gamrnd(0.5*ones(K-1,1),taujl_d); %local variances
logtaujl = log(taujl);

cla = 2;
clb = 8;
cl = 1/gamrnd(cla,1/clb);
logcl = log(cl);

psil = cl*taul*taujl./(cl + taul * taujl); 
beta0 = sqrt(psil) .* randn(K-1,1);


% process SD, v ~ N(0,psi)
tau0 = 1/n;
tau_d = 1/gamrnd(0.5,tau0^2);
tau = 1/gamrnd(0.5, tau_d); %global variance
logtau = log(tau);

tauj_d = 1./gamrnd(0.5,1,K-1,1);
tauj = 1./gamrnd(0.5*ones(K-1,1),tauj_d); %individual variances
logtauj = log(tauj);

cva = 2;
cvb = 8;
cv = 1/gamrnd(cva,1/cvb);
logcv = log(cv);

psi = cv * tau * tauj ./ (cv + tau * tauj);
v = sqrt(psi) .* randn(K-1,1); 


%% Initialize
beta_star = cumsum(randn(n,K-1));

z = zeros(n,1);
for t = 1:n
    tmp = u + x(t,2:K) * beta0 + sum(x(t,2:K)' .* v .* beta_star(t,:)'); 
    zt_mean = tmp;
    zt_std = 1;
    if y(t) > 0
        z(t) = zt_mean + zt_std * trandn(-zt_mean/zt_std,Inf);
    else
        z(t) = zt_mean + zt_std * trandn(-Inf,-zt_mean/zt_std);
    end
end


state_var = cell(n,1);
for t = 1:n
    state_var{t} = eye(K-1);
end
varz = ones(n,1); %variance of epsilon
H = eye(n); % AR slope matrix for z
HH = eye(n); % H'*H
F = sparse(diag(2 * y - 1)); %slope: constraints of z
g = sparse(zeros(n,1)); %intercept: constraints of z


%% Set up adaptive MH 
KK = 2*(K+1);
pstar = 0.25;
tmp_const = -norminv(0.5*pstar);
AMH_c = 1/(KK * pstar * (1-pstar)) + (1-1/KK)*0.5*sqrt(2*pi)*...
    exp(0.5*tmp_const*tmp_const)/tmp_const;
logrw_pv = 0;
logrw_start_pv = logrw_pv;
drawi_start_pv = 0; 
pv_mean = zeros(KK,1);
pv_cov = zeros(KK,KK); %hyper-para for beta0,v



%% MCMC
draws.beta = cell(K-1,1);
for j = 1:K-1
    draws.beta{j} = zeros(ndraws,n);
end %TVP 
draws.z = zeros(ndraws,n);
draws.p1 = zeros(ndraws,n);
draws.corr_eps = zeros(ndraws,1);

draws.alpha0 = zeros(ndraws,K); %[u, beta0]
draws.beta0_d = zeros(ndraws,K+1); %hyper-para for beta0
draws.v = zeros(ndraws,K-1);
draws.v_d = zeros(ndraws,K+1); %hyper-para for v

draws.logrw_pv = zeros(ndraws,1); 
draws.count_pv = 0;
% draws.b = zeros(ndraws,1); %working para for PXDA

if ind_pred == 1
    draws.bn_mean = zeros(ndraws,K-1);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K-1,K-1);
    end
end

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K+1);
    draws.beta0_nonconst_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K+1,1);
    for j = 1:K+1
        draws.beta_sparse{j} = zeros(ndraws,n);
    end
    draws.p1_sparse = zeros(ndraws,n);
end

ntotal = burnin + ndraws;
tic;
for drawi = 1:ntotal   
    % Draw beta_star in (z-SA, beta-AA): simulation smoother
    zstar = z - u - x(:,2:K) * beta0;
    xstar = x(:,2:K) .* repmat(v',n,1);
    beta_star = Simulation_Smoother_DK(zstar, xstar, varz, state_var(2:n),...
        zeros(K-1,1), state_var{1});
    xbeta_star = x(:,2:K).*beta_star;
    x_xbeta_star = [x  xbeta_star];
    
    
    
    % Draw z in (z-SA, beta-AA): truncated normal, integrate out u,beta0,v    
    xx = x_xbeta_star;
    xtimesH = xx'; %xx' * H 
    
    omega = diag(1./[su; psil; psi]) + xx' * xx; 
    omega_z = HH - xtimesH' * (omega \ xtimesH); %precision of z 
    
    M = omega_z; %precision of z
    mu_r = zeros(n,1); %covector of z
    z_new = HMC_exact(F, g, M, mu_r, false, 2, z);
    z = z_new(:,2); %HMC
    if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
        error('zt <0 when yt > 0');
    end      
    
    
    
    % Draw hyper-parameters for v, beta0: integrate out u, beta0, v
    yint = z;
    xint = x_xbeta_star;
    xtimesy = xint' * yint;
    xtimesx = xint' * xint;
    [logtaul, logtaujl, logcl, logtau, logtauj, logcv, count_pv, pv_mean, pv_cov, ...
        logrw_pv, drawi_start_pv, logrw_start_pv] = regularized_HS_int3(xtimesy, xtimesx,...
        logtaul, logtaujl, logcl, logtau, logtauj, logcv, taul0, tau0, cva, cvb, ...
        drawi, burnin, pv_mean, pv_cov, logrw_pv, drawi_start_pv, logrw_start_pv,...
        AMH_c, pstar, su);     
    psi = exp(logcv - log(1 + exp(logcv)*exp(-logtau)*exp(-logtauj)));    
    psil = exp(logcl - log(1 + exp(logcl)*exp(-logtaul)*exp(-logtaujl)));     
    
    

    % Draw u, beta0, v in (z-SA, beta-AA): linear reg
    zg = z;
    xg = x_xbeta_star;
    
    prop_var_inv = diag(1./[su; psil; psi]) + xg' * xg;
    prop_var = prop_var_inv \ eye(2*K-1);
    prop_mean = prop_var * xg' * zg;
    ubeta0v = mvnrnd(prop_mean, prop_var)';
    u = ubeta0v(1); 
    beta0 = ubeta0v(2:K);
    v = ubeta0v(K+1:2*K-1);
    
   
            
    
    % Compute probability
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
    z_mean = u + sum(x(:,2:K) .* beta,2);
    p1 = normcdf(z_mean);
    

    % Compute the first-order autocorrelation of latent index resid
    eps = z - z_mean;
    corr_eps = corr(eps(1:n-1),eps(2:n));    
    
    
    % Sparsify if needed
    if ind_sparse == 1
        xAR = x;
        xbeta_star = xAR .* beta_star;
        v_sparse = SAVS_vector(v, xbeta_star); 
        
        beta0_nonconst_sparse = SAVS_vector(beta0(2:K+1), xAR(:,2:K+1));
        beta0_sparse = [beta0(1); beta0_nonconst_sparse];
        
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);         
    end 
   
    
    % Run Kalman filter if prediction is needed
    if ind_pred == 1
        zz = z - u;
        xx = x(:,2:K);      
        [bn_mean, bn_cov] = TVP_beta_filter(zz, xx, diag(v.^2), beta0);
    end
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K-1
            draws.beta{j}(i,:) = beta(:,j)';
        end
        draws.z(i,:) = z';
        draws.p1(i,:) = p1';
        draws.corr_eps(i) = corr_eps;

        draws.alpha0(i,:) = [u  beta0'];
        draws.beta0_d(i,:) = [logcl  logtaul  logtaujl'];
        draws.v(i,:) = v';
        draws.v_d(i,:) = [logcv  logtau  logtauj'];
        
        draws.logrw_pv(i) = logrw_pv;
        draws.count_pv = draws.count_pv + count_pv/ndraws;      
%         draws.b(i) = b;

        if ind_sparse == 1
            draws.v_sparse(i,:) = v_sparse';
            draws.beta0_nonconst_sparse(i,:) = beta0_nonconst_sparse';
            for j = 1:K+1
                draws.beta_sparse{j}(i,:) = beta_sparse(:,j)';
            end
            draws.p1_sparse(i,:) = normcdf(sum(xAR.*beta_sparse,2));
        end 
        
        if ind_pred == 1
            draws.bn_mean(i,:) = bn_mean';
            draws.bn_cov{i} = bn_cov;
        end          
    end
    
    
    % Display elapsed time
    if (drawi/5000) == round(drawi/5000)
        disp([num2str(drawi), ' out of ', num2str(ntotal),' draws have completed!']);
        toc;
    end
end







