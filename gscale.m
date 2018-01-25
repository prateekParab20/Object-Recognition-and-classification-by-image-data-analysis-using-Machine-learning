indir = uigetdir(cd, 'Select input folder');
outdir = uigetdir(cd, 'Select output folder');
directory = dir([indir, '\', '*.jpg']);

for i = 1 : length(directory)
    filename = directory(i).name;
    rgb_img = imread([indir, '\', filename]);    
    if (ndims(rgb_img) == 3) %Make sure img is RGB (not gray).
        img = rgb2gray(rgb_img);
        %Save gray image to outdir (keep original name).
        imwrite(img, [outdir, '\', filename]);
    end
end