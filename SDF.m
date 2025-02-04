function [sdf , kernel] = SDF(psth,ftype,w,varargin)

notext = 0;
if nargin>3
    notext = strcmp(varargin{1},'notext');
end

% do the convolution
sdf = psth(:,end);   % preallocate for speed

switch ftype
    
    case 'boxcar'
        if ~mod(w,2)
            w = w + 1;
        end
        kernel      = ones(w,1)/(w);
        sdf         = conv(sdf,kernel,'same');
        [~,maxpos]  = max(kernel);
        kernel      = [(-(w-1)/2:(w-1)/2)',kernel];

    case 'Gauss'
        Gauss_width = max([11 6*w+1]); % hm, should be an odd number... e.g. 11
        kernel      = normpdf(-floor(Gauss_width/2):floor(Gauss_width/2),0,w);
%         kernel      = normpdf(-200:200,0,2)'*normpdf(-1000:1000,0,100);
        sdf         = conv(sdf,kernel,'same');
        [~,maxpos]  = max(kernel);
        kernel      = [-maxpos+1:length(kernel)-maxpos;kernel]';

        % this modification yields the same results as the previous code:
        % dummy       = conv(sdf,kernel);
        % sdf         = dummy(floor(Gauss_width/2)+1:end-floor(Gauss_width/2)); % mode of Gaussian centered on spike -> noncausal

    case 'exp'
        filtbase   = 1:min([3000 5*w]);
        filtfunc   = exp(-filtbase/w);
        kernel     = filtfunc./sum(filtfunc); % set integral to 1
        dummy      = conv(sdf,kernel);
        [~,maxpos] = max(kernel);
        kernel     = [-maxpos+1:length(kernel)-maxpos;kernel]';
        sdf        = dummy(1:size(psth,1));
        
    case 'exGauss'
        if numel(w)~=2
            disp('distributions not fully specified')
            return
        end
        filtbase = (0-3*w(1)):(5*w(2));
        gauss = normpdf(filtbase,0,w(1));
        expo  = exppdf(filtbase,w(2));
        kernel = conv(gauss,expo);
        [~,maxpos] = max(kernel);
        dummy  = conv(sdf,kernel);
        sdf    = dummy(maxpos:end-length(kernel)+maxpos);
        kernel = [-maxpos+1:length(kernel)-maxpos;kernel]';
        if sum(kernel(:,2))<0.99
            disp('kernel density sums to less than 0.99, adjust w')
            return
        end
        
end
end