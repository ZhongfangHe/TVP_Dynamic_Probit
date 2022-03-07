% Estimate a dynamic probit model with TVP:
% yt = I{zt > 0} or sign(zt),
% zt = at * ztm1 + xt' * bt + epst,  epst ~ N(0,1),
% ct = ctm1 + etat,  etat ~ N(0, diag(v2)), ct = [at bt]
% c0 ~ N(0, w)
% z0 ~ N(0, s)
% 
% Estimate v={v2, b0, z0}, bt, zt and other hyper-parameters
% differentiate va and vb in v as those associated with at and bt
% 
% Gibbs sampler: p(b*|z,v), p(z|b*,v), p(v|z,b*)
% use (z/eps)|b* to interweave z0 and vb in the 1st asis
% use (b*/b)|z to interweave va and vb in the 2nd asis
%
% In the 2nd asis, integrate out b0 when drawing v2
%
% Spike-slab for v, beta0 (instead of horseshoe)
% ASIS boosting


function draws = Est_DProbTVP_SS(y, x, burnin, ndraws, ind_sparse, ind_pred)
% Inputs:
%   y: a n-by-1 vector of binary target (1/0, or 1/-1),
%   x: a n-by-K matrix of regressors (first column is ones),
%   burnin: a scalar of the number of burn-ins,
%   ndraws: a scalar of the number of draws after burn-in.
%   ind_sparse: a scalar if sparsification for non-constant coefs is needed
% Outputs:
%   draws: a structure of the draws.

[n,K] = size(x);
minNum = 1e-100;
maxNum = 1e100;
c = 1e-6; %spike constant
% c = 1e-10; %spike constant

%% Priors: initial beta, beta0 ~ N(0, taul * diag(phil)), taul, phil are IBs
var_const = 10; %prior variance for constant

% phil_d = 1./gamrnd(0.5,1,K,1);
% phil = 1./gamrnd(0.5*ones(K,1),phil_d); %local variances
% 
% taul_d = 1/gamrnd(0.5,1);
% taul = 1/gamrnd(0.5, taul_d); %global variance

q_beta0 = rand;
ind_beta0 = zeros(K,1);
sprior_beta0 = [5  5]';
s_beta0 = 1./gamrnd(sprior_beta0(1), 1/sprior_beta0(2), K, 1);

psil = [var_const; ind_beta0.*s_beta0 + c*(1-ind_beta0).*s_beta0]; 
% beta0 = sqrt(psil) .* randn(K+1,1);
beta0 = zeros(K+1,1);


%% Priors: initial z0
z0_prior_mean = 0;
z0_prior_var = 25;
% z0 = z0_mean + sqrt(z0_var) * randn;
z0 = 0;



%% Priors: scaling factor for state noise 
% tau_d = 1/gamrnd(0.5,1);
% tau = 1/gamrnd(0.5, tau_d); %global variance
% 
% tauj_d = 1./gamrnd(0.5,1,K+1,1);
% tauj = 1./gamrnd(0.5*ones(K+1,1),tauj_d); %individual variances

q_v = rand;
ind_v = zeros(K+1,1);
sprior_v = [5  5]';
% sprior_v = [0.5  0.5]';
s_v = 1./gamrnd(sprior_v(1), 1/sprior_v(2), K+1, 1);
psi = ind_v.*s_v + c*(1-ind_v).*s_v;

% v = sqrt(psi) .* randn(K+1,1); %scaling factor for state noise
v = 1e-3 * ones(K+1,1);
v2 = v.^2;


%% Initialize
beta = zeros(n,K+1);
z = zeros(n,1);
for t = 1:n
    if y(t) > 0
        z(t) = trandn(0,Inf);
    else
        z(t) = trandn(-Inf,0);
    end
end


state_var = cell(n,1);
for t = 1:n
    state_var{t} = eye(K+1);
end
vary = ones(n,1);


%% MCMC
draws.beta = cell(K+1,1);
for j = 1:K+1
    draws.beta{j} = zeros(ndraws,n);
end %TVP 
draws.z = zeros(ndraws,n);
draws.p1 = zeros(ndraws,n);
draws.z0 = zeros(ndraws,1);

draws.beta0 = zeros(ndraws,K+1);
draws.beta0_ind = zeros(ndraws,K);
draws.beta0_s = zeros(ndraws,K);
draws.beta0_q = zeros(ndraws,1);

draws.v = zeros(ndraws,K+1);
draws.v_ind = zeros(ndraws,K+1);
draws.v_s = zeros(ndraws,K+1);
draws.v_q = zeros(ndraws,1);

if ind_sparse == 1
    draws.v_sparse = zeros(ndraws,K+1);
    draws.beta0_nonconst_sparse = zeros(ndraws,K);
    draws.beta_sparse = cell(K+1,1);
    for j = 1:K+1
        draws.beta_sparse{j} = zeros(ndraws,n);
    end %TVP
    draws.p1_sparse = zeros(ndraws,n);
end
draws.corr_eps = zeros(ndraws,1);
if ind_pred == 1
    draws.bn_mean = zeros(ndraws,K+1);
    draws.bn_cov = cell(ndraws,1);
    for j = 1:ndraws
        draws.bn_cov{j} = zeros(K+1,K+1);
    end
end

ntotal = burnin + ndraws;
tic;
for drawi = 1:ntotal     
    % Draw beta_star in (beta-AA): simulation smoother
    xAR = [x  [z0;z(1:n-1)]]; 
    zstar = z - xAR * beta0;
    xstar = xAR .* repmat(v',n,1);
    beta_star = Simulation_Smoother_DK(zstar, xstar, vary, state_var(2:n),...
        zeros(K+1,1), state_var{1}); 


    % Draw z in (z-SA, beta-AA): truncated normal
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
%     z = update_z_dynamic_probit(y, x, z, z0, beta); %Gibbs
%     z = update_z_dynamic_probit2(y, x, z, z0, beta); %Decorrelated Gibbs
    [H, mu] = TVP_AR_invert(beta(:,K+1));
    F = sparse(diag(2 * y - 1));
    g = sparse(zeros(n,1));
    M = H' * H;
    mu_r = H' * (mu * z0 + sum(x.*beta(:,1:K),2));
    z_new = HMC_exact(F, g, M, mu_r, false, 2, z);
    z = z_new(:,2); %HMC
    if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
        error('zt <0 when yt > 0');
    end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update indictor of spike-slabe prior for v, beta0
    zlag = [z0;z(1:n-1)];
%     [ind_beta0, ind_v] = SpikeSlab_PriorUpdate_ind_probit(z-x(:,1)*beta0(1), ... 
%         [x(:,2:K) zlag], [x  zlag].*beta_star, vary, c, ind_beta0, ind_v,...
%         s_beta0, s_v, q_beta0, q_v);
    [ind_beta0, ind_v] = SpikeSlab_PriorUpdate_ind_probit2(z, ... 
        [x zlag], [x  zlag].*beta_star, vary, c, ind_beta0, ind_v,...
        s_beta0, s_v, q_beta0, q_v, var_const);    
    psi = ind_v.*s_v + c*(1-ind_v).*s_v;
    psil = [var_const; ind_beta0.*s_beta0 + c*(1-ind_beta0).*s_beta0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Draw v,beta0 in (z-SA, beta-AA): linear reg
    xAR = [x  [z0;z(1:n-1)]];
    xx = [xAR  xAR.*beta_star];
    beta0v_prior_cov_inv = diag([1./psil; 1./psi]);
    beta0v_B_inv = beta0v_prior_cov_inv + xx' * xx;
    beta0v_B = beta0v_B_inv\eye(2*K+2);
    beta0v_b = beta0v_B * (xx' * z);
    beta0v = mvnrnd(beta0v_b, beta0v_B)';    
    beta0 = beta0v(1:K+1);
    v = beta0v(K+2:2*K+2);


    % Draw z0 in (z-SA, beta-AA)
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
    z0_post_var_inv = 1/z0_prior_var + beta(1,K+1)^2;
    z0_post_var = 1/z0_post_var_inv;
    z0_post_mean = z0_post_var * beta(1,K+1) * (z(1) - x(1,:) * beta(1,1:K)');
    z0 = z0_post_mean + sqrt(z0_post_var) * randn;
    
    
%*************************************************************************   
    % Compute eps for z-AA, conditional on beta_star
    xAR = [x  [z0;z(1:n-1)]];
    eps = z - sum(xAR.*beta,2);

    
    % Draw z0, a0, va in z-AA, conditional on beta_star
    phi_vec = beta(:,K+1);
    [H, mu] = TVP_AR_invert(phi_vec);
    ubeta0v = H \ eps; %intercept
    
    x_zAA = [x  x.*beta_star(:,1:K)];
    xbeta0v = H \ [x_zAA  mu]; %slope matrix
    
    a0v = [beta0(1:K); v(1:K); z0];
    a0v_prior_mean = zeros(2*K+1,1);
    a0v_prior_var = [var_const; psil(2:K); psi(1:K); z0_prior_var];
    
    for jj = 1:50
        a0v = update_beta_probit_AA(y, xbeta0v, ubeta0v, a0v, ...
            a0v_prior_mean, a0v_prior_var); %Gibbs
    end

% %     F = diag(2 * y - 1) * xbeta0v;
%     F = repmat(2 * y - 1,1,2*K+1) .* xbeta0v;
% %     F = sparse(1:n,1:n,2*y-1) * xbeta0v;
%     g = (2 * y - 1) .* ubeta0v;
%     M = diag(a0v_prior_var);
%     mu_r = a0v_prior_mean;
%     a0v_new = HMC_exact(F, g, M, mu_r, true, 2, a0v);
%     a0v = a0v_new(:,2); %HMC  
     
    beta0(1:K) = a0v(1:K);
    v(1:K) = a0v(K+1:2*K); 
    z0 = a0v(2*K+1);
  
    
    % Compute back z in z-AA, conditional on beta_star
    z = xbeta0v * a0v + ubeta0v;
    if or(max(z(y<=0)) > 0, min(z(y>0)) < 0)
        error('zt <0 when yt > 0');
    end    
%*************************************************************************


%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
    % ASIS in beta-SA: compute beta
    beta = repmat(beta0',n,1) + beta_star .* repmat(v',n,1);
    diff_beta = [beta(1,:)-beta0'; beta(2:n,:) - beta(1:(n-1),:)];
    diff_beta2 = diff_beta.^2;  
    
    
    % ASIS in beta-SA: re-draw v
    vsign = sign(v);
    v2 = v.^2;
    for j = 1:K+1
        vj2_old = v(j)^2;
        aj = psi(j); %vj2~G(0.5,2*aj)
        sum_diff_betaj = sum(diff_beta2(2:n,j)); 
        vj2_new = gigrnd(1-0.5*n, 1/aj, sum_diff_betaj, 1);
        
        logprior_old = -0.5*log(vj2_old) - 0.5 * vj2_old / aj;
        logprior_new = -0.5*log(vj2_new) - 0.5 * vj2_new / aj;
        
        vbeta0j = psil(j);
        loglike_old = -0.5*log(vbeta0j+vj2_old) - 0.5*(beta(1,j)^2)/(vbeta0j+vj2_old) ...
            -0.5*(n-1)*log(vj2_old) - 0.5 * sum_diff_betaj / vj2_old; 
        loglike_new = -0.5*log(vbeta0j+vj2_new) - 0.5*(beta(1,j)^2)/(vbeta0j+vj2_new) ...
            -0.5*(n-1)*log(vj2_new) - 0.5 * sum_diff_betaj / vj2_new; 
        
%         logprop_old = (1-0.5*n)*log(vj2_old) - 0.5*vj2_old/aj - 0.5*sum_diff_betaj/vj2_old;
%         logprop_new = (1-0.5*n)*log(vj2_new) - 0.5*vj2_new/aj - 0.5*sum_diff_betaj/vj2_new;
        logprop_old = -0.5*n*log(vj2_old) - 0.5*vj2_old/aj - 0.5*sum_diff_betaj/vj2_old;
        logprop_new = -0.5*n*log(vj2_new) - 0.5*vj2_new/aj - 0.5*sum_diff_betaj/vj2_new;        
        
        logprob = (logprior_new + loglike_new - logprop_new) - ...
            (logprior_old + loglike_old - logprop_old);
        if log(rand) < logprob
            v2(j) = vj2_new;
        end     
    end
    v = sqrt(v2) .* vsign;
    
    
%     for j = 1:K+1
%         [v2(j),~] = gigrnd(0.5-0.5*n, 1/(tau*tauj(j)), sum(diff_beta2(:,j)), 1);
%         if v2(j) == 0
%             v2(j) = minNum;
%         end 
%     end
%     v = sqrt(v2) .* vsign;
    
    
    % ASIS in beta-SA: re-draw beta0
    B_beta0 = 1./(1./psil + 1./v2);
    b_beta0 = psil .* (beta(1,:)')./(v2 + psil);
    beta0 = b_beta0 + sqrt(B_beta0).*randn(K+1,1);   
    
    % ASIS in beta-SA: compute back beta_star
    beta_star = (beta - repmat(beta0',n,1)) ./ repmat(v',n,1);
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     % Generalized Gibbs: draw working parameter a
%     s_vec = [beta0(1:K);  v(1:K);  z0]; %excluding AR coef
%     s2_vec = s_vec.^2;
%     s_prior_var = [psil(1:K); psi(1:K); z0_prior_var];
%     a_tmp1 = sum(s2_vec./s_prior_var);
%     
%     xAR = [x  [z0;z(1:n-1)]];
%     eps = z - sum(xAR.*beta,2);
%     a_tmp2 = eps' * eps;
%     a_tmp = a_tmp1 + a_tmp2;
%     
%     a2 = gamrnd(0.5*(n+2*K+1), 2/a_tmp);
%     a = sqrt(a2);
%     
%     z = a * z;
%     beta0(1:K) = a * beta0(1:K);
%     v(1:K) = a * v(1:K);
%     z0 = a * z0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
    
    % Update hyperparameters for v
%     [tau, tau_d, tauj, tauj_d] = Horseshoe_update_vector(v2, tau, tau_d, ...
%         tauj, tauj_d); %Horseshoe
%     [ind_v, s_v, q_v] = SpikeSlab_PriorUpdate_cond(v, ind_v, c, sprior_v);
%     psi = ind_v.*s_v + c*(1-ind_v).*s_v; %spike-slab conditional
    [s_v, q_v] = SpikeSlab_PriorUpdate_sq_marg(v, ind_v, c, sprior_v); %spike-slab marginal
    
    % Update hyperparameters of beta0
%     beta02 = beta0(2:K+1).^2; 
%     [taul, taul_d, phil, phil_d] = Horseshoe_update_vector(beta02, taul, taul_d, ...
%         phil, phil_d); %Horseshoe 
%     [ind_beta0, s_beta0, q_beta0] = SpikeSlab_PriorUpdate_cond(beta0(2:K+1), ind_beta0,...
%         c, sprior_beta0);
%     psil = [var_const; ind_beta0.*s_beta0 + c*(1-ind_beta0).*s_beta0]; %spike-slab conditional
    [s_beta0, q_beta0] = SpikeSlab_PriorUpdate_sq_marg(beta0(2:K+1), ind_beta0,...
        c, sprior_beta0); %spike-slab marginal    
    
    % Compute probability
    xAR = [x  [z0;z(1:n-1)]];
    z_mean = sum(xAR.*beta,2);     
    p1 = normcdf(z_mean);
    
    

    % Compute the first-order autocorrelation of latent index resid
    eps = z - z_mean;
    corr_eps = corr(eps(1:n-1),eps(2:n));    
    
    
    % Sparsify if needed
    if ind_sparse == 1
        xbeta_star = xAR .* beta_star;
        v_sparse = SAVS_vector(v, xbeta_star); 
        
        beta0_nonconst_sparse = SAVS_vector(beta0(2:K+1), xAR(:,2:K+1));
        beta0_sparse = [beta0(1); beta0_nonconst_sparse];
        
        beta_sparse = beta_star .* repmat(v_sparse',n,1) + repmat(beta0_sparse',n,1);         
    end 
   
    
    % Run Kalman filter if prediction is needed
    if ind_pred == 1
        [bn_mean, bn_cov] = TVP_beta_filter(z, xAR, diag(v2), beta0);
    end
    

    % Collect draws
    if drawi > burnin
        i = drawi - burnin;
        for j = 1:K+1
            draws.beta{j}(i,:) = beta(:,j)';
        end
        draws.z(i,:) = z';
        draws.p1(i,:) = p1';
        draws.z0(i) = z0;
        draws.corr_eps(i) = corr_eps;

        draws.beta0(i,:) = beta0';
        draws.beta0_ind(i,:) = ind_beta0';
        draws.beta0_s(i,:) = s_beta0';
        draws.beta0_q(i) = q_beta0;

        draws.v(i,:) = v';
        draws.v_ind(i,:) = ind_v';
        draws.v_s(i,:) = s_v';
        draws.v_q(i) = q_v;

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







