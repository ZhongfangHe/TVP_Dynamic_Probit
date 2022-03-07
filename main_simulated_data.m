% Estimate dynamic probit (DProb) and TVP dynamic probit (DProbTVP) models for simulated data


clear;
rng(123456);
dbstop if warning;
dbstop if error;

dgp = {'DProb','DProbTVP'}; %data generating process
ndgp = length(dgp);

for dgpj = 1:ndgp 
    for wsj = 1:10 %10 simulation from each dgp
        % Read data to get y, x
        read_file = ['Simulated_Data_', dgp{dgpj}, '_IB.xlsx'];
        read_sheet = ['D',num2str(wsj)];
        if dgpj == 1 %DProb
            data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:F301');
        else %DProbTVP
            data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:I301');
        end
        y = data(:,1);
        x = data(:,2:4);
        p1true = data(:,6);

        % Setup of estimation
        ind_sparse = 0;
        ind_pred = 0;

        burnin = 1000*2;
        ndraws = 5000*2;
        disp(['burnin = ', num2str(burnin), ', ndraws = ', num2str(ndraws)]);

        % DProb HS
        draws = Est_DProb_HS(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_dprob = sqrt(mean(tmp.^2))';
        xlswrite(read_file, rmse_dprob, read_sheet, 'K2');
        
        % DProb SS
        draws = Est_DProb_SS(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_dprob = sqrt(mean(tmp.^2))';
        xlswrite(read_file, rmse_dprob, read_sheet, 'L2');        
        
        % DProbTVP HS
        draws = Est_DProbTVP_HS(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_dprob = sqrt(mean(tmp.^2))';
        xlswrite(read_file, rmse_dprob, read_sheet, 'M2');
        
        % DProbTVP SS
        draws = Est_DProbTVP_SS(y, x, burnin, ndraws, ind_sparse, ind_pred);
        tmp = draws.p1 - repmat(p1true',ndraws,1); 
        rmse_dprob = sqrt(mean(tmp.^2))';
        xlswrite(read_file, rmse_dprob, read_sheet, 'N2');        
    end
end









