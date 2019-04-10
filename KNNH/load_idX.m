function idX = load_idX(filepath, use_gpu, K, X)
    if (exist(filepath, 'file'))
        fprintf('Start loading precomputed index matrix.\n');
        load(filepath);
        fprintf('Load index matrix done.\n');
    else
        if (use_gpu == 1)
            StepSize = 800; % if you encounter an out of memory error, then decrease stepsize to 400
            % X : [N, d]
            V_GPU = gpuArray(X);
            idX = zeros(size(X,1), K+1);
            n = 0;
            time_consumption = 0;
            for i = 1:size(X,1)/StepSize
                Q_GPU = V_GPU((i-1)*StepSize+1:StepSize*i,:);
                tic;
                idx_GPU = knnsearch(V_GPU, Q_GPU, 'K', K+1, 'Distance', 'euclidean');
                time_consumption = time_consumption + toc;
                tmp_idx = gather(idx_GPU);
                idX((i-1)*StepSize+1:StepSize*i,:) = tmp_idx;
                n = n + StepSize;
                fprintf('KNNH Search : %d th iter has finished.\r', i);
            end
            Q_GPU = V_GPU(n+1:end,:);
            tic;
            idx_GPU = knnsearch(V_GPU, Q_GPU, 'K', K+1, 'Distance', 'euclidean');
            time_consumption = time_consumption + toc
            tmp_idx = gather(idx_GPU);
            idX(n+1:end,:) = tmp_idx;
            fprintf('KNNH Search Done.\n');
        else
            tic;
            idX = knnsearch(X, X, 'K', K+1, 'Distance', 'euclidean');
            toc;
        end
        save(filepath, 'idX', '-v7.3');
    end
    idX = idX(:,1:K);
end