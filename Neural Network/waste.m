p_waste=output_y'-y;
p_waste_2=p_waste;
for m=1:29
   if p_waste(m)<0
       p_waste_2(m)=0;
   end
end
plot(p_waste_2);
hold on
plot(Waste);
