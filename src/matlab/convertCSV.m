
ARR_1 = zeros(1,30);
ARR_2 = zeros(1,30);
ARR_3 = zeros(1,30);
ARR_OUT = zeros(500,90); %% 500줄을 얻고 싶다면 zeros(500,90);

k = 1; %반복(iteration)을 위한 초기화
t = 300; %특정 부분부터 잘라서 가져오고 싶을 때, 최초시작지점 선택

while(k <= 500)
csi_trace = read_bf_file('C:\Users\ACA\Documents\MATLAB\matlab\real_data\standing10.data');
csi_entry = csi_trace{t};
csi = get_scaled_csi(csi_entry);


A = abs(csi);
%B = db(A);

i = 1;
while(i<=30)
   
    ARR_1(i) = A(:,1,i);
    ARR_2(i) = A(:,2,i);
    ARR_3(i) = A(:,3,i);
    i = i + 1;
    
end

ARR_FINAL = [ARR_1,ARR_2,ARR_3]; %합치기
ARR_OUT(k,:) = ARR_FINAL;

disp(k);

k = k + 1; 
t = t + 1;
end
csvwrite('test_stand.csv', ARR_OUT);



