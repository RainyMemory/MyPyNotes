import time
import threading

simulateFolderNum = 0
myLocker = threading.Lock()

def demo_func(num) :
    global simulateFolderNum
    myLocker.acquire()
    simulateFolderNum += 1
    try :
        print(threading.current_thread().name, ": I'm reading folder :", simulateFolderNum, sep=" ")
        simulateFolderNum += num + 1
    finally :
        myLocker.release()

if __name__ == "__main__" :
    for threadNum in range(10) :
        thread = threading.Thread(target=demo_func, args=(threadNum,))
        thread.start()
        thread.join()
    print("Finally we reached folder :", simulateFolderNum, sep=" ")
