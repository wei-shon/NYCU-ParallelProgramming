import matplotlib.pyplot as plt

# 設定圖片大小為長15、寬10
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

Numbers = [ 'filter 1', 'filter 2', 'filter 3']
Cuda = [25.84, 8.52, 17.28]
OpenCL = [30.37, 10.06, 20.65]

# 把資料放進來並指定對應的X軸、Y軸的資料，用方形做標記(s-)，並指定線條顏色為紅色，使用label標記線條含意

plt.plot(Numbers,Cuda,'s-',color = 'r', label="Cuda")
for a,b in zip(Numbers,Cuda):
    plt.text(a, b+0.05, '%.2fx' % b, ha='center', va= 'bottom',fontsize=20)

plt.plot(Numbers,OpenCL,'s-',color = 'b', label="OpenCL")
for a,b in zip(Numbers,OpenCL):
    plt.text(a, b+0.05, '%.2fx' % b, ha='center', va= 'bottom',fontsize=20)


# 設定圖片標題，以及指定字型設定，x代表與圖案最左側的距離，y代表與圖片的距離
plt.title("Cuda v.s. OpenCL", x=0.5, y=1.03, fontsize = 40)
# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 標示x軸(labelpad代表與圖片的距離)
plt.xlabel("fliter", fontsize=30, labelpad = 15)
# 標示y軸(labelpad代表與圖片的距離)
plt.ylabel("Speedup", fontsize=30, labelpad = 20)

# 顯示出線條標記位置
plt.legend(loc = "best", fontsize=25)
# 畫出圖片
plt.savefig("performace.jpg")
plt.show()