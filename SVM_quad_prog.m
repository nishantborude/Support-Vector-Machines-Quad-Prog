load('q3_1_data.mat')

C_ar = [0.1, 10];
for C = C_ar
    mid = trD' * trD;
    H = diag(trLb) * mid * diag(trLb);

    f = -1 * ones(size(trLb));
    A = [];
    b = [];

    Aeq = trLb';
    beq = 0;

    lb = zeros(size(trLb));
    ub = C * ones(size(trLb));

    [alpha_min, min_val] = quadprog(H,f,A,b,Aeq,beq,lb,ub);

    w = trD * (alpha_min.*trLb);
    b = trLb - (trD' * w);
    b = mean(b);

    train_acc = acc_score(w, b, trD, trLb)
    cval_acc = acc_score(w, b, valD, valLb)

    -1 * min_val

    sup_vect = calc_sup_vec(w, b, valD, valLb)

    pred = sign(valD' * w + b);
    c_mat = confusionmat(pred, valLb)
end

function acc = acc_score(weights, bias, data, labels)
    pred = sign(data' * weights + bias);
    acc = 0;
    for i = 1:length(labels)
        if pred(i) == labels(i)
            acc = acc + 1;
        end
    end
    acc = acc * 100/length(labels);
end

function ans = calc_sup_vec(weights, bias, data, labels)
    pred = data' * weights + bias;
    sup = pred <= 1 & pred >= -1;
    ans = nnz(sup);
end
