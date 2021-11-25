for i=1:40
    dirnum=i;
    sdirnum=num2str(i);
    for j=1:10
        filenum=j;
        sfilenum=num2str(j);
        filename=['s',sdirnum,'\',sfilenum,'.pgm'];
        temp=imread(filename);
        newfilename=['s',sdirnum,'\',sfilenum,'.bmp'];
        
        imwrite(temp,newfilename,'bmp');
    end
end