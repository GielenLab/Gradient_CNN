clear all

close all

base_folder='D:\Fabrice\data Exeter\2023-01-13-paper-gradient\Matlab codes for Github\Matlab image processing example\';

sigma=300; %used for flat field correction factor imflatfield function - size of Gaussian filter
total_hist=[]; %mean of fluorescence for all beads

lst_files = dir(fullfile(strcat(base_folder,'detections_cells\'),'*.npz')); % Gets a list of all npz files for cell detections in folder
lst_files_fluo = dir(fullfile(strcat(base_folder,'detections\'),'*.npz')); % Gets a list of all npz files flor fluorescence bead detection in folder
lst_files_im = dir(fullfile(base_folder,'*.tif')); % Gets a list of all tif files in folder

%natural sort order of list_files
[~, Index] = natsort({lst_files.name});
lst_files2   = lst_files(Index);

%natural sort order of list_files
[~, Index] = natsort({lst_files_fluo.name});
lst_files2_fluo   = lst_files_fluo(Index);

%natural sort order of list_files
[~, Index] = natsort({lst_files_im.name});
lst_files_im_sorted   = lst_files_im(Index);


number=zeros(length(lst_files2),1); % total number of beads
fluo=[]; %stores fluorescence values for beads signal MINUS background
center_BB=[];
center_BB_name=[];

offset=0.75; % offset used to take a radius of circle smaller than bounding box  
total=[]; % stores fluo value for all beads

score_thresh=0.5; % used to exclude low confidence detections
spheroid_size_min =250 ; %min size of spheroid 
spheroid_size_max=5000; %max size of spheroid 

size_spheroid=[]; %stores max lengths of spheroids
total_fluo_overall=[];
spheroid_size_overall=[];

for k = 1:length(lst_files2) % loop over each file bounding boxes
   
        %%initializations %%%%%%%%%
        area=[];
        fluo_pos=[];
        spheroid_area=[];
        total_fluo=[];
        spheroid_size=[];
        size_spheroid=[];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%


    full_file_name = fullfile(strcat(base_folder,'detections_cells\'), lst_files2(k).name); % join folderpath and filename
      
    full_file_name_fluo = fullfile(strcat(base_folder,'detections\'), lst_files2_fluo(k).name); % join folderpath and filename
      
    [nb ,bboxes,scores_sph]=read_npy(full_file_name); %spheroid bounding boxes
    [nb_fluo ,bboxes_fluo,score_fluo]=read_npy(full_file_name_fluo); %beads bounding boxes
   
    full_file_name_im = fullfile(base_folder, lst_files_im_sorted(2*k-1).name); % get path to bright field image
    full_file_name_bf = fullfile(base_folder, lst_files_im_sorted(2*k).name); % get path to fluorescence image
   
    if full_file_name_im(end-5:end-4)=='w1'
    
        im_fluo=imread(full_file_name_im,'tif');
        im_bf=imread(full_file_name_bf,'tif');
        im_fluo_corr=imflatfield(im_fluo,sigma); %flat field correction
    
        background_main_fluo=mode(mode(im_fluo_corr));
      
        figure(1)
        imagesc(im_fluo_corr); %im_bf
        colormap gray
        
    end

    spheroid_pos=zeros(floor(nb),4);
    
    %draw rectangles for bounding boxes
    for i=1:floor(nb) % nb is total number of beads detected
        
         length_box=bboxes(i,4)-bboxes(i,2);
         width_box=bboxes(i,3)-bboxes(i,1);

         radius=max(length_box,width_box);
         center_x=bboxes(i,2)+radius/2;
         center_y=bboxes(i,1)+radius/2;
        
         %claculate fluo intensity within each bead using a circular mask           
         [columnsInImage, rowsInImage] = meshgrid(1:size(im_fluo_corr,1), 1:size(im_fluo_corr,2));
                   
                    circlePixels=(rowsInImage - (center_y)).^2 ...
                    + (columnsInImage - (center_x)).^2 <= ((radius.*offset)/2).^2;

                      circlePixels_background=(rowsInImage - (center_y)).^2 ...
                    + (columnsInImage - (center_x)).^2 <= ((radius*1.25)/2).^2;

                                                    
                    bead_masked_raw= double(im_fluo_corr).*double(circlePixels); %apply circular mask
                    
                    bead_masked=bead_masked_raw; %copy for getting mean
                    bead_masked(bead_masked_raw==0) = [];
                    
                    total(i)=mean(mean(bead_masked))./double(background_main_fluo); %mean of bead pixels
                            
        
        spheroid_pos(i,:)=[bboxes(i,2) bboxes(i,1) radius-1 radius-1 ];
        rectangle('Position',spheroid_pos(i,:),'EdgeColor','blue'); hold on;
              
        size_spheroid=[size_spheroid;length_box.*width_box];

       end
        
        %%Show Fluo beads detections
        fluo_pos=zeros(round(nb_fluo),4);

        for i=1:nb_fluo
        
            radius=max(bboxes_fluo(i,4)-bboxes_fluo(i,2),bboxes_fluo(i,3)-bboxes_fluo(i,1)) ;
            
            hold on;    
            fluo_pos(i,:)=[bboxes_fluo(i,2) bboxes_fluo(i,1) radius-1 radius-1 ];
            rectangle('Position',fluo_pos(i,:),'EdgeColor','g'); hold on; %draw rectangle around beads
        
        end

%%%%%%%  Find which spheroids correspond to which fluo bead %%%%%%%%%%%%%%%

area = rectint(spheroid_pos,fluo_pos); 

if length(area)~=0

            %divide areas by spheroid area to see if 100% contained in fluo bead
        
        for i=1:size(area,1)
        
            for j=1:size(area,2)
        
            spheroid_area(i,j)=size_spheroid(i);
            end
        
        end
        
        area_norm=area./spheroid_area*100; %get overlap in percentage
        
        %%%%find indices for which spheroids have corresponding fluo bead
        fluo_index=zeros(1,size(area_norm,1));
        
        for i=1:size(area_norm,1)
        
            for j=1:size(area,2)
            
                if area_norm(i,j)>90 && scores_sph(i)>score_thresh %matching pair
            
                    fluo_index(i)=j;
                   
                end
            
            end
        
        end

                     
        
        for i=1:size(area_norm,1)
        
            if fluo_index(i)~=0
                
                fluo_pos_local=fluo_pos(fluo_index(i),:); % position as components x y width height 
                min_size=min(fluo_pos_local(3),fluo_pos_local(4));
        
                fluo_bead_bf=imcrop(im_bf, fluo_pos_local);
                size_mask= min(min(size(fluo_bead_bf),min_size));
                fluo_bead_bf=imresize(fluo_bead_bf, [min_size min_size]);
                
                
                fluo_bead_fluo=imcrop(im_fluo_corr, fluo_pos_local);
                fluo_bead_fluo=imresize(fluo_bead_fluo, [min_size min_size]);
               
                %%here mask outside and take mean inside
                
                center_x=fluo_pos_local(2)+size_mask/2;
                center_y=fluo_pos_local(1)+size_mask/2;
        
                 [columnsInImage, rowsInImage] = meshgrid(1:size_mask, 1:size_mask);
                           
                            circlePixels=(rowsInImage - (size_mask/2)).^2 ...
                            + (columnsInImage - (size_mask/2)).^2 <= ((size_mask.*offset)/2).^2;
        
        
                    %%%%  ignore beads which are at the edges %%%%%%%%%%%%%
                    if (size(fluo_bead_fluo,1)==size(circlePixels,1)-1 || size(fluo_bead_fluo,1)==size(circlePixels,1))  && size(fluo_bead_fluo,2)==size(circlePixels,2)


                        fluo_bead_fluo = imresize(fluo_bead_fluo,[size(circlePixels,1) size(circlePixels,2)]); %resize in case there is a 1 pixel difference
                   
                        spheroid_masked=double(fluo_bead_fluo).*double(circlePixels); %crop spheroid
                        local_background=double(fluo_bead_fluo).*double(not(circlePixels)); %local background around beads

                        spheroid_masked(spheroid_masked==0) = NaN;
                        local_background(local_background==0) = NaN;
                        
                        total_fluo(i)=double(nanmean(nanmean(spheroid_masked)))- double(nanmean(nanmean(local_background))); %total fluorescence minus background from every bead
                        
                        spheroid_size(i)=spheroid_area(i,1);

                    end
                
            end
        
        end
        
       % [~, Index] = sort(total_fluo); % sort according to bead intensity
        %l=0;

       
end

total_fluo_overall=[total_fluo_overall total_fluo];
spheroid_size_overall=[spheroid_size_overall spheroid_size];


end

%remove zeros from arrays

Index_0=(total_fluo_overall==0);
spheroid_size_overall(Index_0==1)=[];
total_fluo_overall(Index_0==1)=[];

%plot spheroid area versus fluorescence
figure(2)
scatter(total_fluo_overall,spheroid_size_overall);
ylim([0 3000]);
xlabel('Fluorescence Intensity (A.U.)','FontSize',16, 'FontWeight', 'bold');
ylabel('Spheroid estimated area (\mum^2)','FontSize',16, 'FontWeight', 'bold');


%%%%%%%%%%%%%%%%% READING NUMPIES %%%%%%%%%%%%%%%%%%%%%
function [nz,bboxes,scores] =read_npy(full_file_name)
   
    a=unzip(full_file_name); 

    b=readNPY(a{1}); %read numpies 
    scores=readNPY(a{2}); %get scores
    b=squeeze(b); %remove first dimension which is not used
    nz=nnz(b)/4 ;
    bboxes=b;
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%