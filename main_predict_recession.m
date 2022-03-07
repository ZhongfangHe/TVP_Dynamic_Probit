% Estimate models for recession data

clear;
rng(12345);
dbstop if warning;
dbstop if error;

for h = 1:2 %forecast horizon in quarters
    disp(['forecast horizon h = ', num2str(h)]);


    % Model list
    md = {'DHS','PTHS','DTHS','DTHSdc','DSS','PTSS','DTSS'};
    for mj = [1 5] %select a model
        disp(['Model = ', md{mj}]);


        %% Read data to get y, x
        read_file = 'Data_Recession.xlsx';
        read_sheet = 'Data_for_Estimation'; %change of TB3m
        data_y = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'B2:B272'); 
        data_x = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'C2:D272'); %chg of TB3m, 10y-3m

        data = [data_y  data_x];
        [ntotal,ndata] = size(data);
        y = data(h+1:ntotal, 1); %recession indicator
        x = data(1:ntotal-h, 2:ndata); %lagged regressors


        %% Configuration
        npred = 69; %number of predictions >= 0
        if npred > 0
            ind_pred = 1;
        else
            ind_pred = 0;
        end

        ind_sparse = 0;

        if ismember(mj, [3 4 7]) %TVP dynamic probit
            if h > 1
                ind_sim = 1; %if simulate beta_tph in multi-horizon prediction or set beta_tph = beta_t
            end
        end



        %% Set the size of estimation/prediction sample
        [n,nx] = size(x); %non-constant regressors
        if npred > 0
            nest = n + 1 - h - npred; %number of estimation data
        else
            nest = n;
        end
        disp(['nobs = ', num2str(n), ', nx (non-constant) = ', num2str(nx)]);
        disp(['nest = ', num2str(nest), ', npred = ', num2str(npred)]);


        %% MCMC
        burnin = 2000; 
        ndraws = 5000*2; 
        disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);

        if npred == 0 %in-sample estimation only   
            % Prepare data
            yest = y;
            xest = [ones(n,1)  normalize_data(x)];

            % Estimate
            tic;
            switch mj
                case 1 %dynamic probit, HS
                    draws = Est_DProb_HS(yest, xest, burnin, ndraws, 1, ind_sparse);
                case 2 %TVP probit, HS
                    draws = Est_ProbTVP_HS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                case 3 %TVP dynamic probit, HS
                    draws = Est_DProbTVP_HS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                case 4 %TVP dynamic probit, HS, double-cauchy           
                    draws = Est_DProbTVP_HS_dc(yest, xest, burnin, ndraws, ind_sparse, ind_pred);            
                case 5 %dynamic probit, SS
                    draws = Est_DProb_SS(yest, xest, burnin, ndraws, 1, ind_sparse);
                case 6 %TVP probit, SS
                    draws = Est_ProbTVP_SS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                case 7 %TVP dynamic probit, SS           
                    draws = Est_DProbTVP_SS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);         
            end
            disp('Estimation is finished!');
            save(['Est_', md{mj},'_h', num2str(h),'.mat'],'draws');
            toc;
        else %out-of-sample forecasts
            logpredlike = zeros(npred,2);     
            for predi = 1:npred 
                % Get the estimation sample data
                nesti = nest + predi - 1;
                yi = y(1:nesti,:);
                xi = x(1:nesti,:); 

                % Prepare data
                yest = yi;
                xest = [ones(nesti,1)  normalize_data(xi)];      

                % estimate the model
                switch mj
                    case 1 %dynamic probit, HS
                        draws = Est_DProb_HS(yest, xest, burnin, ndraws, 1, ind_sparse);
                    case 2 %TVP probit, HS
                        draws = Est_ProbTVP_HS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                    case 3 %TVP dynamic probit, HS
                        draws = Est_DProbTVP_HS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                    case 4 %TVP dynamic probit, HS, double-cauchy           
                        draws = Est_DProbTVP_HS_dc(yest, xest, burnin, ndraws, ind_sparse, ind_pred);            
                    case 5 %dynamic probit, SS
                        draws = Est_DProb_SS(yest, xest, burnin, ndraws, 1, ind_sparse);
                    case 6 %TVP probit, SS
                        draws = Est_ProbTVP_SS(yest, xest, burnin, ndraws, ind_sparse, ind_pred);
                    case 7 %TVP dynamic probit, SS           
                        draws = Est_DProbTVP_SS(yest, xest, burnin, ndraws, ind_sparse, ind_pred); 
                end      

                % prediction
                xmean = mean(xi)';
                xstd = std(xi)';

                xtph = [ones(h,1)  (x(nesti+1:nesti+h,:) - repmat(xmean',h,1))...
                    ./repmat(xstd',h,1)];           
                ytph = y(nesti+h);

                if ismember(mj, [1 5]) %dynamic probit
                    if h > 1
                        [ytph_pdf, ytph_pdf_vec] = Pred_DProbit_MultiH(draws, xtph, ytph, h);
                    else
                        [ytph_pdf, ytph_pdf_vec] = Pred_DProbit(draws, xtph', ytph);
                    end
                elseif ismember(mj, [3 4 7]) %TVP dynamic probit
                    if h > 1
                        [ytph_pdf, ytph_pdf_vec] = Pred_DProbTVP_MultiH(draws, xtph, ytph, h, ind_sim);
                    else 
                        [ytph_pdf, ytph_pdf_vec] = Pred_DProbTVP(draws, xtph', ytph);
                    end
                else %TVP probit
                    if h > 1
                        [ytph_pdf, ytph_pdf_vec] = Pred_ProbTVP_MultiH(draws, xtph(h,:)', ytph, h);
                    else 
                        [ytph_pdf, ytph_pdf_vec] = Pred_ProbTVP(draws, xtph', ytph);
                    end            
                end

                % store log likelihoods
                logpredlike(predi,1) = log(ytph_pdf(1))'; 
                disp(['Prediction ', num2str(predi), ' out of ', num2str(npred), ' is finished!']);
                toc;   
                disp(' ');
            end 
            save(['Pred_', md{mj},'_h', num2str(h),'.mat'],'draws','ytph_pdf_vec');

            write_column = {'C','D','E','F','G','H','I'};
            writematrix(logpredlike(:,1), read_file, 'Sheet', ['PLH',num2str(h)],...
                'Range', [write_column{mj},'2']);
        end
    end
end









