import time
print('me')
for x in range(10):
    print("Progress" + str(x), end="\r")
    time.sleep(0.1)
#print("")
