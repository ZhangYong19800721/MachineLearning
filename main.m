clear all;
close all;
format compact;

SURF = ConstructSurfData('D:\imagebase\','*.jpg');
VisualVocabulary_Y = ConstructVisualVocabulary(2048,SURF,5);

save('VisualVocabulary_Y.mat',VisualVocabulary_Y);
