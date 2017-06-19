clear all;
close all;
format compact;

SURF = ConstructSurfData_Color('D:\imagebase\','*.jpeg');
VisualVocabulary_UV = ConstructVisualVocabulary(124,SURF,5);

save('VisualVocabulary_UV.mat',VisualVocabulary_UV);
