close all
clear all

base_folder='D:\Lewis\detections\';

lst_files = dir(fullfile(base_folder,'*.npz')); % Gets a list of all tif files in folder

%natural sort order of list_files
[~, Index] = natsort({lst_files.name});
lst_files2   = lst_files(Index);

number=zeros(length(lst_files2),1); % total number of cells
class_out=zeros(length(lst_files2),1); % class out-of-plane = 1 in class array
class_in=zeros(length(lst_files2),1); % class out-of-plane = 1 in class array

for k = 1:length(lst_files2) % loop over each file

    waitbar(k/length(lst_files2));

    full_file_name = fullfile(base_folder, lst_files2(k).name); % join folderpath and filename
    
    [nb, class]=read_npy(full_file_name);
    number(k)=nb;
    class_out(k)=sum(class~=0);
    class_in(k)=number(k)-class_out(k);
end

figure(1)
plot(number,'ro-');
hold on;
plot(class_out,'bo-');
hold on;
plot(number-class_out,'go-');
legend('total nb','out-of-plane','in-plane');


%%trials at cell counting 
%take average of 12 slices
nb_slices=15;
avg_cell=zeros(length(number)/nb_slices,1);
avg_out=zeros(length(number)/nb_slices,1);
avg_in=zeros(length(number)/nb_slices,1);

for i=1:16

    avg_cell(i)=mean(number((i-1)*nb_slices+1:i*nb_slices));
    avg_out(i)=mean(class_out((i-1)*nb_slices+1:i*nb_slices));
    avg_in(i)=mean(class_in((i-1)*nb_slices+1:i*nb_slices));
    
end

figure(2)
plot(movmean(avg_cell,10),'ro-'); hold on;
plot(movmean(avg_out,10),'bo-'); hold on;
plot(movmean(avg_in,10),'go-');
legend('average total','avg OUT','avg IN');

% %%%%from a single slice as is usually done %%%%%
% cell3=zeros(length(number)/nb_slices,1);
% out3=zeros(length(number)/nb_slices,1);
% in3=zeros(length(number)/nb_slices,1);
% 
% for i=1:201
% 
%     cell3(i)=number((i-1)*nb_slices+3);
%     out3(i)=class_out((i-1)*nb_slices+3);
%     in3(i)=class_in((i-1)*nb_slices+3);
%     
% end
% 
% 
% figure(3)
% 
% plot(cell3,'ro-'); hold on;
% plot(out3,'bo-'); hold on;
% plot(in3,'go-');
% legend('total slice 3','OUT slice 3','IN slice 3');


%%%%%%%%%%%%%%%%% HOW TO READ NUMPIES EXAMPLE %%%%%%%%%%%%%%%%%%%%%
function [nz,class] =read_npy(file_name)
   
    a=unzip(file_name); 

    b=readNPY(a{1}); %read numpies 
    scores=readNPY(a{2});
    b=squeeze(b); %emove first dimension which is not used
    nz=nnz(b)/4 ;%number of non zero elements,divide by 4 coordinates = number of bacteria in total
    class=readNPY(a{3});
end
%%%%%%%%%%%%%%%%%%%%%%%%%%