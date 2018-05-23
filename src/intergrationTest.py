import matlab.engine
import numpy as np
eng = matlab.engine.start_matlab()
#eng.Untitled(nargout=0)
#ARR_1 = eng.zeros(1,30);
#ARR_2 = eng.zeros(1,30);
#ARR_3 = eng.zeros(1,30);

ARR_FINAL = np.empty([0,90], float)
k = 1; #%반복(iteration)을 위한 초기화
t = 0; #%특정 부분부터 잘라서 가져오고 싶을 때, 최초시작지점 선택
csi_trace = eng.read_bf_file('C:/Users/user/Documents/data/walk/walk_04/2018_05_09_walk10_04_delay1000.dat')

while(k <= 500):
    csi_entry = csi_trace[t]
    csi = eng.get_scaled_csi(csi_entry)
    A = eng.abs(csi)
    ARR_OUT = np.empty([0], float)

    # ARR_1 = []
    # ARR_2 = []
    # ARR_3 = []
    ARR_OUT = np.concatenate((ARR_OUT, A[0][0]), axis=0)
    ARR_OUT = np.concatenate((ARR_OUT, A[0][1]), axis=0)
    ARR_OUT = np.concatenate((ARR_OUT, A[0][2]), axis=0)
    # ARR_3.append(A[0][2][:])
    # ARR_FINAL = [ARR_1 + ARR_2 + ARR_3] #% 합치기

    ARR_FINAL = np.vstack((ARR_FINAL, ARR_OUT))
    k = k + 1
    t = t + 1
print(ARR_FINAL)

