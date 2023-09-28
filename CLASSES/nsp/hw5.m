%{ 
Name: Kimberly Nestor
Class: Neural Signal Processing
Problem: HW5
Program goal: Implement k-means algorithm on neural spike snip data.
%}


% load data 
data = importdata('ps5_data.mat');
x = data.('RealWaveform');

% centroids
k2_cen_1 = data.('InitTwoClusters_1');
k2_cen_2 = data.('InitTwoClusters_2');
k3_cen_2 = data.('InitThreeClusters_2');
k3_cen_1 = data.('InitThreeClusters_1');

% params 
tot_sec = 10;
spike_time = linspace(1,10,length(x));
thres = 250;


%%%% Q1 - data pre-processing
% plot original data
% figure;
figure('visible','off');
plot(spike_time, x);
title('Unfiltered data');
xlabel('Time (s, rate = 30kHz)');
ylabel('μV');

% code section from ps5
% x = RealWaveform;
f_0 = 30000; % sampling rate of waveform (Hz) 
f_stop = 250; % stop frequency (Hz)
f_Nyquist = f_0/2; % the Nyquist limit
n = length(x);
f_all = linspace(-f_Nyquist,f_Nyquist,n);
desired_response = ones(n,1);
desired_response(abs(f_all)<=f_stop) = 0;
x_filtered = real(ifft(fft(x).*fftshift(desired_response)));


% plot filtered data
% figure;
figure('visible','off');
plot(spike_time, x_filtered);
title('Filtered data');
xlabel('Time (s, rate = 30kHz)');
ylabel('μV');

% turn on for neuron sounds
% sound(x*.97/max(abs(x)),f_0);


%% Q2 - spike detection
% plot threshold line across high-pass filter data
% figure;
figure('visible','off');
plot(spike_time, x_filtered);
title('Filtered data - threshold = 250');
xlabel('Time (s, rate = 30kHz)');
ylabel('μV');
yline(thres, 'r', 'LineWidth',2);



%% find first values of wave that go above threshold
thres_idx = find(x_filtered > 250);

curr = 1;
thres_idx_cross = [];
for i = 1:length(thres_idx)
    if thres_idx(i) - curr > 1
    thres_idx_cross = [thres_idx_cross; thres_idx(i)];
    end
    curr = thres_idx(i);
end



%% set params
n_waves = length(thres_idx_cross);   %%%%%
% n_waves = 50; %%%% change this
len_waves = (thres_idx_cross(1)+15) - (thres_idx_cross(1)-15) +1 ;

per_s = n/tot_sec; % spikes per 1s
per_ms = per_s/1000; % spikes per 1ms
per_sub_ms = per_ms/10; % spikes per 0.1ms


%% subplot of threshold crossing waves
% figure;
figure('visible','off');
tiledlayout('flow')

wave_snips = [];
for i = 1:n_waves % 6, all
    
    start = thres_idx_cross(i)-10;
    stop = thres_idx_cross(i)+20;

    nexttile
    p = plot(spike_time, x_filtered, '-o', 'MarkerSize',2, 'LineWidth',1); % ms= 3, 2
    xlabel('Time (s');
    ylabel('μV');
    yline(thres, 'r', 'LineWidth',1); % 1 % 2
    xlim([spike_time(start) spike_time(stop)]);
    p.MarkerFaceColor = [1 0.5 0];
    p.MarkerEdgeColor = [1 0.5 0];
    
    snip = x_filtered(start : stop);
    wave_snips = [wave_snips; [snip]];

end
wave_snips = reshape(wave_snips, [len_waves, n_waves]); 



%% Q3 - clustering with K-means, InitTwoClusters_1

% plot original centroids and datapoints
% figure;
figure('visible','off');
plot(k2_cen_1, 'r','DisplayName','centroids');
yline(thres, 'k', 'LineWidth', 1);
hold on
plot(wave_snips, 'b','DisplayName','data')
title('Original centroids');
xlabel('Samples');
ylabel('μV'); % voltage

% get centroid and class assignments
[kclass, cost, centroid] = kmeans_wave(wave_snips, k2_cen_1);


% plot new centroids and data with class assignment
% figure;
figure('visible','off');
plot(centroid, 'r','DisplayName','centroids', 'LineWidth', 5);
yline(thres, 'k', 'LineWidth', 1);

hold on
plot(wave_snips(:,find(kclass == 1)), 'b', 'LineWidth', 0.5);


hold on
plot(wave_snips(:,find(kclass == 2)), 'c', 'LineWidth',0.5);

title('K-means centroids - clusted waves');
xlabel('Samples');
ylabel('μV'); 


% plot decreasing cost function
% figure;
figure('visible','off');
plot(cost);
xlabel('Epoch');
ylabel('Cost'); 


%% Q4 - clustering with K-means, InitTwoClusters_2

% plot original centroids and datapoints
% figure;
figure('visible','off');
plot(k2_cen_2, 'r','DisplayName','centroids');
yline(thres, 'k', 'LineWidth', 1);
hold on
plot(wave_snips(:,1:6), 'b','DisplayName','data')
title('Original centroids');
xlabel('Samples');
ylabel('μV'); % voltage

% get centroid and class assignments
[kclass, cost, centroid] = kmeans_wave(wave_snips, k2_cen_2);


% plot new centroids and data with class assignment
% figure;
figure('visible','off');
plot(centroid, 'r','DisplayName','centroids', 'LineWidth', 5);
yline(thres, 'k', 'LineWidth', 1);

hold on
plot(wave_snips(:,find(kclass == 1)), 'b', 'LineWidth', 0.5);


hold on
plot(wave_snips(:,find(kclass == 2)), 'c', 'LineWidth',0.5);

title('K-means centroids - clusted waves');
xlabel('Samples');
ylabel('μV'); 


% plot decreasing cost function
% figure;
figure('visible','off');
plot(cost);
xlabel('Epoch');
ylabel('Cost');


%% Q5A - clustering with K-means, InitThreeClusters_1

% plot original centroids and datapoints
% figure;
figure('visible','off');
plot(k3_cen_1, 'r','DisplayName','centroids');
yline(thres, 'k', 'LineWidth', 1);
hold on
plot(wave_snips(:,1:6), 'b','DisplayName','data')
title('Original centroids');
xlabel('Samples');
ylabel('μV'); % voltage

% get centroid and class assignments
[kclass, cost, centroid] = kmeans_wave(wave_snips, k3_cen_1);


% plot new centroids and data with class assignment
% figure;
figure('visible','off');
plot(centroid, 'r','DisplayName','centroids', 'LineWidth', 5);
yline(thres, 'k', 'LineWidth', 1);

hold on
plot(wave_snips(:,find(kclass == 1)), 'b', 'LineWidth', 0.5);

hold on
plot(wave_snips(:,find(kclass == 2)), 'c', 'LineWidth',0.5);

hold on
plot(wave_snips(:,find(kclass == 3)), 'g', 'LineWidth',0.5);

title('K-means centroids - clusted waves');
xlabel('Samples');
ylabel('μV'); 


% plot decreasing cost function
% figure;
figure('visible','off');
plot(cost);
xlabel('Epoch');
ylabel('Cost');


%% Q5B - clustering with K-means, InitThreeClusters_2

% plot original centroids and datapoints
% figure;
figure('visible','off');
plot(k3_cen_2, 'r','DisplayName','centroids');
yline(thres, 'k', 'LineWidth', 1);
hold on
plot(wave_snips(:,1:6), 'b','DisplayName','data')
title('Original centroids');
xlabel('Samples');
ylabel('μV'); % voltage

% get centroid and class assignments
[kclass, cost, centroid] = kmeans_wave(wave_snips, k3_cen_2);


% plot new centroids and data with class assignment
% figure;
figure('visible','off');
plot(centroid, 'r','DisplayName','centroids', 'LineWidth', 5);
yline(thres, 'k', 'LineWidth', 1);

hold on
plot(wave_snips(:,find(kclass == 1)), 'b', 'LineWidth', 0.5);

hold on
plot(wave_snips(:,find(kclass == 2)), 'c', 'LineWidth',0.5);

hold on
plot(wave_snips(:,find(kclass == 3)), 'g', 'LineWidth',0.5);

title('K-means centroids - clusted waves');
xlabel('Samples');
ylabel('μV'); 


% plot decreasing cost function
% figure;
figure('visible','off');
plot(cost);
xlabel('Epoch');
ylabel('Cost');


%%
function [k_class, j_func, cen] = kmeans_wave(wave_mat, cen)
    % k-means function on a vector waveform
    j_func = [];
    k_class_all = [];

    dim_mat = size(wave_mat);
    dim_cen = size(cen);
    
    for ii = 1:1000 % each epoch
    
        k_class = [];
        k_dist = [];
        for i = 1:dim_mat(2) % each data point, find shortest distance to centroid
            
            cen_dist = [];
            for k = 1:dim_cen(2) % each centroid
                dist = norm(wave_mat(:,i) - cen(:,k));
                cen_dist = [cen_dist; dist^2];
    
            end
            [k_min, k_idx] = min(cen_dist);
    
            k_class = [k_class; k_idx];
            k_dist = [k_dist; k_min];
        end

        % assign classes and obj func j for curr epoch
        j_func = [j_func; sum(k_dist)];
        k_class_all = [k_class_all; {k_class}];
        
        % find updated centroid
        k_lst = unique(k_class);
        
        cen = [];
        for iii = 1:length(k_lst)
        cen_idx = find(k_class == iii);

        mu_k = mean(wave_mat(:, cen_idx), 2);
        
        cen = [cen; mu_k];
        end
        cen = reshape(cen, dim_cen); 
        

        % check tolerance
        if ii >2 & isequal(k_class, k_class_all{end}) & isequal(k_class, k_class_all{end-1})
            break
        end
    end
end









