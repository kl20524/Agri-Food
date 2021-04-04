daily_week=zeros(30,7);
for k=1:30
M=sold(k); 
i=rand(1,7); 
j=M*i/(sum(i));
nev=round(j); %
nev(7) = M-sum(nev(1:6));
daily_week(k,:)=nev;
end

x=daily_week(1:29,:);
y=sold(2:30);




