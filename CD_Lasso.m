function [w,b] = CD_Lasso(y, X, lambda)
    % y is n*1 vector
    % X is d*n vector
    
    % initialize lasso
    [d,n]=size(X);
    
    w = (X*X'+lambda)\(X*y);
%   w = (X*X')\(X*y);
    a = 2*sum(X.^2, 2);
    b = sum(y-X'*w)/n;
    c = zeros(d,1);
    delta=1e-6;
    solution=0;
    iteration=0;
    
    disp('I will iteration 100 times if not converged during process ...');
    
    % coodinate descent
    while(iteration<100) 
        w_old=w;
        b_old=b;
        
        r=y-X'*w_old-b_old;
        b=sum(r)/n+b_old;
        r=r+b_old-b;
        
        for k=1:d
            Xi = X(k,:);
            c(k,1)=2*Xi*(r+Xi'*w(k));
            
            if(c(k,1)< -lambda)
                w(k)=(c(k,1)+lambda)/a(k);
            elseif(c(k,1) > lambda)
                w(k)=(c(k,1)-lambda)/a(k);
            else
                w(k)=0;
            end
            r = r - (Xi'*(w(k) - w_old(k)));
        end

        solution_old=solution;
        solution =sum((X'*w+b-y).^2)+lambda*sum(abs(w));
        
        if(abs(solution-solution_old) <= delta)
            break;
        end
        
        iteration=iteration+1;
        disp(['iteration: ',num2str(iteration)]);
        
    end
    
end



