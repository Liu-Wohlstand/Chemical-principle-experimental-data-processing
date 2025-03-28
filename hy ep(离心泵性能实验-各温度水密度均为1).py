import matplotlib.pyplot as plt
import numpy as np
import csv
import random
#一般将实验者姓名作为.csv的命名，如:李华做了实验则泵性能文件命名为lihua.csv，管性能命名为lihua2.csv
bengtexingceding = 'lihua.csv'#离心泵性能曲线
guanlutexingceding = 'lihua2.csv'#管路性能曲线
k = 77.914 * 1000  # 单位为 “次每升” *1000后为次每立方米
#---以下请勿改动---------------------------------------------------

ls = []
with open(bengtexingceding, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        str_data = ','.join(row)
        str_data = str_data.replace('\ufeff', '')
        result = str_data.split(',')
        float_row = [float(item) for item in result]
        ls.append(float_row)

#  ls: ls[0]流量计 ls[1]压力表 ls[2]真空表 ls[3]电机功率 ls[4]水温度
g = 9.8  # 单位为m/(s^2)

ls_f_pinlvji = []
ls_p_ru = []
ls_p_chu = []
ls_T_H2O = []
ls_P = []  # 电机功率

for i in ls:
    ls_f_pinlvji.append(i[0])
    ls_p_chu.append(i[1])
    ls_p_ru.append(-i[2])
    ls_T_H2O.append(i[4])
    ls_P.append(i[3])








d_guanlu = 0.040
d_zhenkongbiao = 0.025
d_yalibiao = 0.025


def yangcheng(p_ru, p_chu, rou_H2O, f_pinlvji):
    Q = 4 / 3600  # 离心泵流量 4 m^3/h
    #  扬程计算公式 H=(z_chu-z_ru)+(p_chu-p_ru)/(rou_H2O*g)+(u_chu**2-u_ru**2)/(2*g)
    vs_chu = f_pinlvji / (k)  # 出流量
    #  流速计算公式 u=vs/A
    A_guanlu = np.pi * (d_guanlu / 2) ** 2
    A_yalibiao = np.pi * (d_yalibiao / 2) ** 2  # 压力表处管内切面面积
    A_zhenkongbiao = np.pi * (d_zhenkongbiao / 2) ** 2  # 真空表处管内切面面积
    u_ru = Q / A_guanlu
    H = 0.18 + (p_chu - p_ru) / (rou_H2O * 1000 * g)
    Ne = H * vs_chu * rou_H2O * 9.8

    print(f'出流量：{vs_chu*3600:.2f} m^3/h')
    #  print(f'管路截面面积：{A_guanlu} m^2')
    #  print(f'真空表处截面面积：{A_zhenkongbiao} m^2')
    #  print(f'进液流速=出液流速={u_ru} m/s')
    print(f'扬程为{H:.2f} m')
    print(f'有效功率为{Ne:.3f} kW')

    return H, vs_chu, Ne, A_guanlu


ls_H = []
ls_vs_chu = []
ls_Ne = []
#  轴功率
ls_P_shuru = np.array(ls_P) * 0.6

for i in range(22):
    print(f'管路性能 第{i + 1}组实验处理数据如下：')

    print(f'第{i + 1}组实验处理数据如下：')
    H, vs_chu, Ne, A_guanlu = yangcheng(ls_p_ru[i], ls_p_chu[i], 1, ls_f_pinlvji[i])
    ls_H.append(H)
    ls_vs_chu.append(vs_chu)
    ls_Ne.append(Ne)
    print(f'轴功率为{ls_P_shuru[i]:.3f} kW')
    print()
print(f'管路面积为{A_guanlu} m^2')
ls_H = np.array(ls_H)
ls_vs_chu = np.array(ls_vs_chu) * 3600
print()
#  效率
yita = np.array(ls_Ne) / (np.array(ls_P) * 0.6)
for i in range(len(yita)):
    print(f'第{i+1}组实验效率为{yita[i]:.2f}')
print()
def quxiannihehanshu1(x, y):
    p = np.polyfit(x, y, 2)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(p, x_fit)
    return x_fit, y_fit


#  管路性能
ls2 = []
with open(guanlutexingceding, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        str_data = ','.join(row)
        str_data = str_data.replace('\ufeff', '')
        result = str_data.split(',')
        float_row = [float(item) for item in result]
        ls2.append(float_row)

#  ls2: ls[0]泵频率 ls[1]压力表 ls[2]真空表 ls[3]电机功率 ls[4]水温度 ls[5]流量计
ls_f_pinlvji2 = []
ls_p_ru2 = []
ls_p_chu2 = []
ls_T_H2O2 = []
ls_P2 = []  # 电机功率
ls_f_beng = []
for i in ls2:
    ls_f_beng.append(i[0])
    ls_f_pinlvji2.append(i[5])
    ls_p_chu2.append(i[1])
    ls_p_ru2.append(-i[2])
    ls_T_H2O2.append(i[4])
    ls_P2.append(i[3])

ls_H2 = []
ls_vs_chu_2 = []
for i in range(len(ls_p_chu2)):
    print(f'管路性能 第{i + 1}组实验处理数据如下：')
    H2, vs_chu2, Ne2, A_guanlu2 = yangcheng(ls_p_ru2[i], ls_p_chu2[i], 1, ls_f_pinlvji2[i])
    ls_H2.append(H2)
    ls_vs_chu_2.append(vs_chu2)
    print()
ls_vs_chu_2 = np.array(ls_vs_chu_2) * 3600
seed1 = random.randint(0, len(ls))
print(f"选择泵性能测定实验第{seed1}组数据\n"
      f"计算过程如下：\n"
      f"该组数据水温为{ls_T_H2O[seed1]}℃，该温度下水密度ρ_水={1}*10^3 kg/m^3\n"
      f"u_出=u_入=Q/A_真空表=Q/A=Q/(0.25*d^2*π)\n"
      f"        =(4/3600)/(0.25*{d_guanlu}^2*π)\n"
      f"        ={A_guanlu} m/s\n"
      f"\n"
      f"Q=f_频率计/k={ls_f_pinlvji[seed1]}/{k}={ls_vs_chu[seed1]/3600} m^3/s\n"
      f"\n"
      f"H=(z_出-z_入)+(p_出-p_入)/(ρ_水*g)+(u_出**2-u_入**2)/(2*g)\n"
      f" =(z_出-z_入)+(p_压力表-p_真空表)/(ρ_水*g)+(u_出**2-u_入**2)/(2*g)#如果真空表读数加负号则压差为“-”，如果无负号则为“+”\n"
      f" =0.18+({ls_p_chu[seed1]}-{-ls_p_ru[seed1]})/({1}*9.8*1000)\n"
      f" ={ls_H[seed1]} m\n"
      f"\n"
      f"Ne=H*Q*g*ρ_水/1000={ls_H[seed1]}*{ls_vs_chu[seed1]}*9.8*{1}/1000={ls_Ne[seed1]} kW"
      f"\n"
      f"N=P*0.6={ls_P[seed1]}*0.6={ls_P_shuru[seed1]} kW"
      f"\n"
      f"η=Ne/N={ls_Ne[seed1]}/{ls_P_shuru[seed1]}={yita[seed1]}"
      f"\n")
seed1 = random.randint(0, len(ls2))
print(f"选择管道性能测定实验第{seed1}组数据\n"
      f"计算过程如下：\n"
      f"该组数据水温为{ls_T_H2O2[seed1]}℃，该温度下水密度ρ_水={1}*10^3 kg/m^3\n"
      f"u_出=u_入=Q/A_真空表=Q/A=Q/(0.25*d^2*π)\n"
      f"        =(4/3600)/(0.25*{d_guanlu}^2*π)\n"
      f"        ={A_guanlu} m/s\n"
      
      f"\n"
      f"H=(z_出-z_入)+(p_出-p_入)/(ρ_水*g)+(u_出**2-u_入**2)/(2*g)\n"
      f" =(z_出-z_入)+(p_压力表-p_真空表)/(ρ_水*g)+(u_出**2-u_入**2)/(2*g)#如果真空表读数加负号则压差为“-”，如果无负号则为“+”\n"
      f" =0.18+({ls_p_chu2[seed1]}-{-ls_p_ru2[seed1]})/({1}*9.8*1000)\n"
      f" ={ls_H2[seed1]} m\n")

# 拟合
quxian_vs_chu, quxian_H = quxiannihehanshu1(ls_vs_chu, ls_H)
print(ls_P_shuru)
quxian_vs_chu2, quxian_P = quxiannihehanshu1(ls_vs_chu, ls_P_shuru)

quxian_vs_chu3, quxian_yita = quxiannihehanshu1(ls_vs_chu, yita)

quxian_vs_chu4, quxian_H_2 = quxiannihehanshu1(ls_vs_chu_2, ls_H2)
#  绘图
width_cm = 16.02

width_inch = width_cm / 2.54
height_inch = width_cm/ 2.54


fig, ax1 = plt.subplots(figsize=(width_inch, height_inch))

ax1.plot(quxian_vs_chu, quxian_H, color='green', label='H',linestyle='-')
ax1.scatter(ls_vs_chu, ls_H, color='green', marker='x')
ax1.plot(quxian_vs_chu4, quxian_H_2, label='He',linestyle='--')
ax1.scatter(ls_vs_chu_2, ls_H2, marker='x')
ax1.set_xlabel('Q (m^3/h)')
ax1.set_ylabel('H (m)')

ax2 = ax1.twinx()
ax2.plot(quxian_vs_chu2, quxian_P, 'b-', label='P',linestyle='-.')
ax2.scatter(ls_vs_chu, ls_P_shuru, color='blue', marker='x')
ax2.plot(quxian_vs_chu3, quxian_yita, color='orange', label='η',linestyle=':')
ax2.scatter(ls_vs_chu, yita, color='orange', marker='x')
ax2.set_ylabel('P (kW)  η ')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_xlim(0,10)
ax1.set_ylim(0,25)
ax2.set_ylim(0,1)
plt.show()
