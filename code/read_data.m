%load hourly_GPS_1.04.mat

all_data = [id TIME LAT LON DROGUE GAP LAT_ERR LON_ERR U V U_ERR V_ERR U_ERR_R V_ERR_R];
%filter locations of data
cond1 = all_data(:,3)>55 & all_data(:,3) < 65;
cond2 = all_data(:,4)>295 & all_data(:,4) < 315;
cond = cond1 & cond2;

all_data(~cond,:) = [];
%%
%filter the drifters without drogue
cond3 = all_data(:,5) == 0;
all_data(cond3,:)= [];
%%
writematrix(all_data,'Data_Filtered')