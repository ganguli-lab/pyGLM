% make plot for cosyne
% Niru Maheswaranathan
% 09:40 PM Nov 18, 2013

load trueParams;
load estParams;

dt = 0.1;
time = 0:dt:dt*(size(data.n,1)-1);

% plot RFs
%neurons = [1 8 56 59 64 68 70 71 80 83 87];
neurons = [59 70 80 83];

% make firings matrix
spkmat = flipud(data.n');
spkmat = spkmat - 2;
firings = spikes2firings(spkmat, time);

ni = firings(:,2) > 80;
ne = firings(:,2) <= 80;


%fig(1); clf; hold on;
%plot(firings(ne,1), firings(ne,2), 'bo'); %, 'FillColor', [0.23 0.14 0.7], 'MarkerSize', 18);
%plot(firings(ni,1), firings(ni,2), 'ro'); %, 'Color', [0.8 0.1 0.14], 'MarkerSize', 18);
%xlim([90 210]); makepretty;
%set(gca,'XTick',90:20:210,'XTickLabel',0:20:120);
%xlabel('Time (s)');
%ylabel('Neuron Index');
%legend('Excitatory', 'Inhibitory', 'Location', 'EastOutside')

% correlation coefficients
%C = corrcoef([theta.w thetaEst.w]);
%C = C(101:end,1:100).^2;

%fig(2)

W0 = zeros(256,4);
W1 = zeros(256,4);
idx = 1;

for n = neurons

    %subplot(121);
    im0 = imsq(theta.w(:,n))';
    %imgsc(im0);
    %title('true')
    %subplot(122);

    W0(:,idx) = im0(:);

    im = imsq(thetaEst.w(:,n))';
    H = fspecial('gaussian', [10 10], 1);
    imf = imfilter(im, H);

    if im0(:)'*imf(:) < 0
        imf = -imf;
    end

    W1(:,idx) = imf(:);

    %imgsc(imf);
    %title('est')

    %mtit(['neuron ' num2str(n)]);
    %waitforbuttonpress;

    idx = idx +1;

end
