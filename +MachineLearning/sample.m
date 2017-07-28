function y = sample(x)
% SAMPLE sample a binary state according to the input probability
    y = (rand(size(x))<x);
end

