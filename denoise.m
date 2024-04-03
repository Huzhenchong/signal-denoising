%% 卡尔曼滤波
[y,fs]=audioread("task.wav"); 
%y返回为 m×n 矩阵，其中 m 是读取的音频样本数，n 是文件中的音频通道数。fs为采样率，单位为赫兹

%只取一声道
y = y(:,1);
%sound(y,fs);

% 画出原始语音信号的时域波形及频谱
t = (0:length(y)-1)/fs; 
subplot(3,2,1);
plot(t, y);
title('原始语音信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Y = fft(y);
f = (0:length(Y)-1)*fs/length(Y);
subplot(3,2,2);
plot(f, abs(Y));
title('原始语音信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');

% 叠加高斯白噪声
noise=0.01*randn(size(y)); 
% 叠加高频噪声
%noise = 0.05 * sin(2*pi*10000*t); % 频率为10000Hz的高频噪声
%noise = noise';

noisy_y = y + noise; 
%sound(noisy_y,fs);
audiowrite('C:\Users\1342053313\Desktop\Digital Signal Processing\denoise\noised.wav',noisy_y,fs);
subplot(3,2,3);
plot(t, noisy_y);
title('加噪信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Noisy_Y = fft(noisy_y);
subplot(3,2,4);
plot(f, abs(Noisy_Y));
title('加噪信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');

% 设定卡尔曼滤波器参数
F = 1; % 状态转移矩阵
H = 1; % 观测矩阵
Q = 0.01; % 状态噪声方差
R = 0.1; % 观测噪声方差
x = 0; % 初始状态估计
P = 1; % 初始估计误差方差

% 初始化滤波后函数
filtered_y = zeros(size(noisy_y));

% 卡尔曼滤波处理
for i = 1:length(noisy_y)
    % 预测步骤
    x = F * x; % 状态预测
    P = F * P * F' + Q; % 估计误差预测
    
    % 更新步骤
    K = P * H' / (H * P * H' + R); % 卡尔曼增益
    x = x + K * (noisy_y(i) - H * x); % 状态更新
    P = (1 - K * H) * P; % 估计误差更新
    
    % 存储滤波后的信号
    filtered_y(i) = x;
end



% 使用设计好的滤波器对加噪信号进行滤波
denoised_y = filtered_y; % 使用卡尔曼滤波器得到的去噪信号

% 画出去噪信号的时域及频谱图
subplot(3,2,5);
plot(t, denoised_y);
title('去噪信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Denoised_Y = fft(denoised_y);
subplot(3,2,6);
plot(f, abs(Denoised_Y));
title('去噪信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');

% 计算原始音频信号的信号能量
original_signal_energy = sum(y.^2);

% 计算去噪后的音频信号的信号能量
denoised_signal_energy = sum(filtered_y.^2);


% 计算噪声信号的能量
noise_energy = sum(noise.^2);

% 计算滤波后的噪声能量
noise2 = y-filtered_y;
noise_energy2 = sum(noise2.^2);


% 计算信噪比（SNR）
SNR_before = 10 * log10(original_signal_energy / noise_energy);
SNR_after = 10 * log10(denoised_signal_energy / noise_energy2);

% 显示信噪比（SNR）
disp(['信噪比（SNR）- 去噪前: ', num2str(SNR_before), ' dB']);
disp(['信噪比（SNR）- 去噪后: ', num2str(SNR_after), ' dB']);

%sound(denoised_y,fs);
audiowrite('C:\Users\1342053313\Desktop\Digital Signal Processing\denoise\denoised.wav',denoised_y,fs);

%% 巴特沃斯低通滤波器
[y,fs]=audioread("task.wav");
%只取一声道
y = y(:,1);
% 画出原始语音信号的时域波形及频谱
t = (0:length(y)-1)/fs; 
subplot(3,2,1);
plot(t, y);
title('原始语音信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Y = fft(y);
%频率
f = (0:length(Y)-1)*fs/length(Y);
subplot(3,2,2);
plot(f, abs(Y));
title('原始语音信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');

% 叠加高斯白噪声
%noise=0.01*randn(size(y)); 

% 叠加高频噪声
noise = 0.05 * sin(2*pi*10000*t); % 频率为10000Hz的高频噪声
noise = noise';


noisy_y = y + noise; 
audiowrite('C:\Users\1342053313\Desktop\Digital Signal Processing\denoise\noised(high frequency).wav',noisy_y,fs);
subplot(3,2,3);
plot(t, noisy_y);
title('加噪信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Noisy_Y = fft(noisy_y);
subplot(3,2,4);
plot(f, abs(Noisy_Y(:,1)));
title('加噪信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');

% 使用巴特沃斯滤波器对加噪信号进行滤波
cutoff_frequency = 6000; % 截止频率
order = 4; % 滤波器阶数
[b, a] = butter(order, cutoff_frequency / (fs/2), 'low');
%filtfilt 函数会首先使用给定的滤波器系数对输入信号进行正向滤波，然后再用相同的滤波器系数对得到的信号进行反向滤波。
denoised_y = filtfilt(b, a, noisy_y); 

% 画出去噪信号的时域及频谱图
subplot(3,2,5);
plot(t, denoised_y);
title('去噪信号的时域波形');
xlabel('时间(s)');
ylabel('幅度');
Denoised_Y = fft(denoised_y);
subplot(3,2,6);
plot(f, abs(Denoised_Y));
title('去噪信号的频谱');
xlabel('频率(Hz)');
ylabel('幅度');
audiowrite('C:\Users\1342053313\Desktop\Digital Signal Processing\denoise\denoised(high frequency).wav',denoised_y,fs);
% 计算原始音频信号的信号能量
original_signal_energy = sum(y.^2);

% 计算去噪后的音频信号的信号能量
denoised_signal_energy = sum(denoised_y.^2);

% 计算噪声信号的能量
noise_energy = sum(noise.^2);

% 计算滤波后的噪声能量
noise2 = y - denoised_y;
noise_energy2 = sum(noise2.^2);

% 计算信噪比（SNR）
SNR_before = 10 * log10(original_signal_energy / noise_energy);
SNR_after = 10 * log10(denoised_signal_energy / noise_energy2);

% 显示信噪比（SNR）
disp(['信噪比（SNR）- 去噪前: ', num2str(SNR_before), ' dB']);
disp(['信噪比（SNR）- 去噪后: ', num2str(SNR_after), ' dB']);