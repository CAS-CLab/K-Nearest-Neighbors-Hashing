function use_gpu = check_gpu(gpuDevice_ID)
    if (gpuDevice_ID < 0)
        use_gpu = 0;
        return;
    end
    
    try
        gpuDevice(gpuDevice_ID);
    catch
        use_gpu = 0;
        return;
    end
    
    a = ver;
    i = 1;
    while ( (i <= size(a,2)) && (~strcmp(a(i).Name,'MATLAB')) )
        i = i + 1;
    end

    idx = find(a(i).Version == '.');

    v1 = str2double(a(i).Version(1:idx-1));
    v2 = str2double(a(i).Version(idx+1:end));

    if ( (v1 >= 9) && (v2 >= 1) ) % MatLab 9.1 == MatLab R2016b
        use_gpu = 1;
    else
        if ( v1 > 9 )
            use_gpu = 1;
        else
            use_gpu = 0;
        end
    end
end