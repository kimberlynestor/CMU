%{ 
Name: Kimberly Nestor
Class: Neural Signal Processing
Problem: HW7
Program goal: Implement PCA dimensionality reduction on neural spike data. From d=31 to d=2.
%}


% load data 
data = importdata('ps7_data.mat');

% init theta params - mu, Sigma, pi - for EM
init = data.('InitParams');

% spike snippets
spike_snips = data.('Spikes');


%% Q1 - plot original spikes
%%%% 1A - plot original datapoints
% figure;
figure('visible','off');
plot(spike_snips, 'b','DisplayName','data')
title('Original data');
xlabel('Sample Dimension');
ylabel('Amplitude (μV)'); % voltage


%%%%% 1B - plot principle component waveforms for first 3
[scores, comps, vals] = pca_spikes(spike_snips, 3);

% figure;
figure('visible','off'); 

plot(comps(1,:), 'Color', '#A2142F', 'DisplayName', 'PC1', 'LineWidth', 2);

hold on
plot(comps(2,:), 'Color', '#77AC30', 'DisplayName', 'PC2', 'LineWidth', 2);

hold on
plot(comps(3,:).* -1, 'Color', '#0072BD', 'DisplayName', 'PC3', 'LineWidth', 2);

xlabel('Sample Dimension', 'FontSize', 15);
ylabel('Amplitude (μV)', 'FontSize', 15); % voltage
title('First three principle components');

lgd = legend;
lgd.FontSize = 14;


%%%%% 1C - plot sqrt eigen_vals - check elbow
% figure;
figure('visible','off'); 

p = plot(sqrt(vals), '-o', 'MarkerSize', 5, 'Color', '#7E2F8E', 'LineWidth', 2);
xlabel('Sample Dimension', 'FontSize', 15);
ylabel('√λ', 'FontSize', 15); 
title('Square-rooted eigenvalue spectrum');

p.MarkerFaceColor = '#7E2F8E';
p.MarkerEdgeColor = '#7E2F8E';


%%%% 1D - plot scatter of pc1 and pc2 scores
% figure;
figure('visible','off');
scatter(scores(1,:).* -1, scores(2,:), 10, 'filled', 'MarkerEdgeColor','#D95319', 'MarkerFaceColor','#D95319')
xlabel('PC score 1', 'FontSize', 15);
ylabel('PC score 2', 'FontSize', 15); 



%% Q2 - run EM GMM on 2D pca clustersn mm
addpath('/Users/kimberlynestor/Desktop/CMU/CLASSES/2023spring/NSP/Homework/hw6') 

k = 3;

init.mu = init.mu(:, 1:k);
init.sigma = repmat(init.Sigma, 1, 1, k);
init.pi = repmat(1/k, 1, k);

scores_sign = reshape([scores(1,:).* -1, scores(2,:)], 2,[]); % flip sign

[em_mu, em_sig, em_pik, liho] = func_GMM(init, scores_sign);
em_pik

figure;
plot(liho)

%%
function [pc_scores, p_comps, eig_vals] = pca_spikes(wave_mat, k)
    % pca function for dimensionality reduction
    dim_mat = size(wave_mat);
    
    % normlaize data, remove mean
    org_mean = mean(wave_mat, 1);
    wave_mat_norm = wave_mat - repmat(org_mean, dim_mat(1), 1);
    
    % find covariance and eigen vec and vals
    spike_cov = cov(wave_mat_norm);
    [eig_vecs, eig_vals] = eig(spike_cov);
    
    eig_vals = fliplr(diag(eig_vals).');
    eig_vecs = fliplr(eig_vecs);

    % find k max eig_vals 
    [~,k_idxs] = maxk(eig_vals, k);


    % loop to find p_comps and pc_scores
    p_comps = [];
    pc_scores = [];
    for i = 1:length(k_idxs) % each k

        % principle components
        comp = sum(wave_mat_norm .* repmat(eig_vecs(:,k_idxs(i)).', dim_mat(1), 1), 2) ;
        
        % pc scores - reduced data
        score = sum(wave_mat_norm .* repmat(comp, 1, dim_mat(2)), 1) ;
        
        % add comps and scores to lsts
        p_comps = [p_comps; [comp.']]; 
        pc_scores = [pc_scores; [score]]; 

    end 
    eig_vals = eig_vals(1:dim_mat(1));
end


