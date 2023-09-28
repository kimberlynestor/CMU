%{ 
Name: Kimberly Nestor
Class: Neural Signal Processing
Problem: HW6
Program goal: Implement the EM algorithm on neural spike snip data.
%}


% load data 
data = importdata('ps6_data.mat');

% init theta params - mu, Sigma, pi
init_1 = data.('InitParams1');
init_2 = data.('InitParams2');

% spike snippets
spike_snips = data.('Spikes');


%% Q2 - implement EM algorithm, GMM, InitParams1

% plot original centroids and datapoints
% figure;
figure('visible','off');
plot(spike_snips, 'b','DisplayName','data')
title('Original centroids');
xlabel('Samples');
ylabel('μV'); % voltage

hold on
plot(init_1.mu, 'r','DisplayName','centroids', 'LineWidth', 2);

% run EM and plot log likelihood
[kclass, mu, sigma, prior, liho] = em_gmm(spike_snips, init_1);

mu_std = cell2mat(arrayfun(@(k) sqrt(diag(sigma(:,:,k))), 1:3,'UniformOutput',false));


% figure;
figure('visible','off');
plot(liho);
xlabel('Epoch')
ylabel('Log likelihood')

% prior param estimate, k= 1,2,3
prior

% plot new centroids and data with class assignment
% figure;
figure('visible','off');

plot(spike_snips(:,find(kclass == 1)), 'b', 'LineWidth', 0.5);

hold on
plot(spike_snips(:,find(kclass == 2)), 'm', 'LineWidth',0.5);

hold on
plot(spike_snips(:,find(kclass == 3)), 'c', 'LineWidth',0.5);

% hold on
% plot(mu, 'r','DisplayName','centroids', 'LineWidth', 2);

% hold on
% plot(mu + mu_std , 'r', 'DisplayName','centroids', 'LineWidth', 2, 'LineStyle', ':');

% hold on
% plot(mu - mu_std , 'r', 'DisplayName','centroids', 'LineWidth', 2, 'LineStyle', ':');

% title('EM centroids - clusted waves');
xlabel('Samples');
ylabel('μV'); 


%% Q3 - implement EM algorithm, GMM, InitParams2
[kclass, mu, sigma, prior, liho] = em_gmm(spike_snips, init_2);


%%
function [k_class, mu_k, sigma_k, pi_k, likelihood] = em_gmm(wave_mat, init_theta)
    % em function on a vector waveform
    mu_k = init_theta.('mu'); % cluster mean, mu_k
    pi_k = init_theta.('pi'); % prior prob, p(z=k)

    dim_mat = size(wave_mat);
    dim_cen = size(mu_k);

    sigma_k = repmat(init_theta.('Sigma'),1,1, dim_cen(2));

    tot_n = dim_mat(2);
    k_lst = 1:dim_cen(2);

    likelihood = []; 
    for ii = 1:100 % each epoch
        k_class = []; 
        likeli_nall = []; 
        xall_gamma = []; 
        gamma_nk_all = []; 
        for i = 1:dim_mat(2) % each data point
            x_zall = [];
            for k = 1:dim_cen(2) % each k, find p(x|z=k)
                x_z = mvnpdf(wave_mat(:,i), mu_k(:,k), sigma_k(:,:,k)) * pi_k(k); % log
                x_zall = [x_zall; x_z];   
            end

            % find probs for datapoint
            joint = x_zall.';
            gamma_nk = joint ./ sum(joint, 2);
            likeli_n = sum(joint .* gamma_nk, 2);
            
            % update gamma_nk
            xn_gamma = gamma_nk .* wave_mat(:,i);
            xall_gamma = [xall_gamma; [xn_gamma]]; 
            gamma_nk_all = [gamma_nk_all; [gamma_nk]]; 
             
            % datapoint assigned class
            [~,idx] = max(gamma_nk);
            k_class = [k_class; idx];
            likeli_nall = [likeli_nall; likeli_n]; 

        end
        e_likeli = sum(likeli_nall, 1);
        likelihood = [likelihood; log(e_likeli)];
        
        % find updated params
        n_k = histc(k_class, k_lst).';
        
        pi_k = (n_k/tot_n);
        mu_k = sum(reshape(xall_gamma, [], length(k_lst), tot_n), 3) ./ n_k;
        
        % loop to update sigma_k
        sig_nall_kall = []; 
        for k = 1:dim_cen(2)
            sig_k = zeros(31,31);
            for i = 1:dim_mat(2)
                sig_k = sig_k + gamma_nk_all(i,k) * ((wave_mat(:,i) - mu_k(:,k)) * (wave_mat(:,i) - mu_k(:,k)).');
            end
            sig_kall = sig_k ./ n_k(k);
            sig_nall_kall(:,:,k) = sig_kall; 

        end
        sigma_k = sig_nall_kall;        
    end
end

