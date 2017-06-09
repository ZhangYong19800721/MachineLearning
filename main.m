clear all;
close all;

SURF = ConstructSurfData('D:\imagebase\','*.jpeg');

VV = ConstructVisualVocabulary(2048,SURF);