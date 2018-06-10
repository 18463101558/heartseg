#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

def draw(loss,dice_acc, accuracy,val_loss,val_dice_acc,val_accuracy):
    plt.figure(2)
    plt.plot(np.arange(len(loss)),loss,label='loss')
    plt.plot(np.arange(len(dice_acc)),dice_acc,label='dice_acc')
    plt.plot(np.arange(len(accuracy)),accuracy,label='accuracy')
    plt.plot(np.arange(len(val_loss)),val_loss,label='val_loss')
    plt.plot(np.arange(len(val_dice_acc)),val_dice_acc,label='val_dice_acc')
    plt.plot(np.arange(len(val_accuracy)),val_accuracy,label='val_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('train-accuracy')
    plt.legend()
    plt.title('The training process')
    plt.show()
def read_file():
    f = open("losshistory.txt")
    losses = []
    dice_acces = []
    val_losses=[]
    accuracyes=[]
    val_dice_acces=[]
    val_accuracyes=[]
    epoches=[]
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        epoch,loss,dice_acc, accuracy,val_loss,val_dice_acc,val_accuracy=line.split(",")
        epoches.append(epoch)
        losses.append(float(loss))
        dice_acces.append(float(dice_acc))
        accuracyes.append(float(accuracy))
        val_losses.append(float(val_loss))
        val_dice_acces.append(float(val_dice_acc))
        val_accuracyes.append(float(val_accuracy))
        line = f.readline()
    f.close()
    draw(losses,dice_acces, accuracyes,val_losses,val_dice_acces,val_accuracyes)
    print("drawed")
if __name__ == '__main__':
    read_file()