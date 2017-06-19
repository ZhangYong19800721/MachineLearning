clear all;
close all;
format compact;

SURF = ConstructSurfData('D:\imagebase\','*.jpeg');
VV = ConstructVisualVocabulary(2048,SURF,5);
