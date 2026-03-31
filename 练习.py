x1=input().split(".")
y1=[]
x2=bin(int(input()))[2:].zfill(32)
y2=[]
res=[]
for i in x1:
    if 0<=int(i)<=255:
        y1.append(bin(int(i))[2:].zfill(8))
        z1="".join(y1)
for i in range(0,32,8):
    y2.append(int(x2[i:i+8],2))
    z2=".".join(str(i) for i in y2)
print(int(z1,2))
print(z2)


