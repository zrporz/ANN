import json
import matplotlib.pyplot as plt
import os
import os.path as osp

def plot_one_experiment(name):
    name = name.split('.')[0] 
    with open(f"{name}.json","r") as f:
        log = json.load(f)
    train_loss = log['train_loss']
    train_loss = [train_loss[i] for i in range(len(train_loss)) if i%12==0]
    test_loss = log['test_loss']
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # 创建一个图表
    # plt.figure(figsize=(10, 6))
    ax1.plot(train_num_epoch, train_loss,  linestyle='-', color='b', label='Train Loss')
    ax1.plot(test_num_epoch, test_loss,  linestyle='-', color='r', label='Test Loss')
    try:
        title = name.split('_')[5]
    except:
        title=""
    ax1.set_title(f"{title}_Loss ")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    # plt.savefig(f"{name}_loss.png")
    # plt.show()

    train_acc = log['train_acc']
    train_acc = [train_acc[i] for i in range(len(train_acc)) if i%12==0]
    test_acc = log['test_acc']
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]

    # 创建一个图表
    # plt.figure(figsize=(10, 6))
    ax2.plot(train_num_epoch, train_acc, linestyle='-', color='b', label='Train accuracy')
    ax2.plot(test_num_epoch, test_acc, linestyle='-', color='r', label='Test accuracy')
    ax2.set_title(f'{title}_Accuracy ')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(" Accuracy ")
    ax2.grid(True)

    # Add legend to each subplot
    ax1.legend()
    ax2.legend()

    # Adjust layout
    plt.tight_layout()


    plt.savefig(f"{name}_acc_loss.png")
    # plt.show()

def plot_contrast(name1,name2):
    name = "contrast"
    log1 = json.load(open(name1))
    log2 = json.load(open(name2))
    train_num_epoch = range(100)
    train_loss1 = [log1['train_loss'][i] for i in range(len(log1['train_loss'])) if i%12==0]
    train_loss2 = [log2['train_loss'][i] for i in range(len(log2['train_loss'])) if i%12==0]
    train_acc1 = [log1['train_acc'][i] for i in range(len(log1['train_acc'])) if i%12==0]
    train_acc2 = [log2['train_acc'][i] for i in range(len(log2['train_acc'])) if i%12==0]
    test_acc1 = log1['test_acc']
    test_acc2 = log2['test_acc']
    test_loss1 = log1['test_loss']
    test_loss2 = log2['test_loss']
    train_num_epoch = range(5,100)
    test_num_epoch = [i for i in range(5,100) if i%5==0]
    train_loss1 = train_loss1[5:]
    train_loss2 = train_loss2[5:]
    train_acc1 = train_acc1[5:]
    train_acc2 = train_acc2[5:]
    test_loss1 = test_loss1[1:]
    test_loss2 = test_loss2[1:]
    test_acc1 = test_acc1[1:]
    test_acc2 = test_acc2[1:]
    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.plot(train_num_epoch, train_loss1,  linestyle='-', color='b', label='Train Loss 1')
    ax1.plot(train_num_epoch, train_loss2,  linestyle='-', color='r', label='Train Loss (DropOut)')
    ax1.plot(test_num_epoch, test_loss1,  linestyle='-', color='cornflowerblue', label='Test Loss 1')
    ax1.plot(test_num_epoch, test_loss2,  linestyle='-', color='tomato', label='Test Loss (DropOut)')
    ax2.plot(train_num_epoch, train_acc1,  linestyle='-', color='b', label='Train Accuracy 1')
    ax2.plot(train_num_epoch, train_acc2,  linestyle='-', color='r', label='Train Accuracy (DropOut)')
    ax2.plot(test_num_epoch, test_acc1,  linestyle='-', color='cornflowerblue', label='Test Accuracy 1')
    ax2.plot(test_num_epoch, test_acc2,  linestyle='-', color='tomato', label='Test Accuracy (DropOut)')
     # Add legend to each subplot
    ax1.legend()
    ax2.legend()

    # Adjust layout
    plt.tight_layout()


    plt.savefig(f"{name}_acc_loss.png")

def plot_contrast_activation(dir):
    
    name_list = os.listdir(dir)
    name_list = [name for name in name_list if name.endswith('.json')]
    act_name = [name.split('_')[1] for name in name_list]
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for name in name_list:
        path = osp.join(dir,name)
        with open(path,"r") as f:
            log = json.load(f)
            train_loss.append([log['train_loss'][i] for i in range(len(log['train_loss'])) if i%12==0])
            train_acc.append([log['train_acc'][i] for i in range(len(log['train_acc'])) if i%12==0])
            test_loss.append(log['test_loss'])
            test_acc.append(log['test_acc'])
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]

    fig, axes= plt.subplots(3, 4, figsize=(20, 12))

    # 创建一个图表
    # plt.figure(figsize=(10, 6))
    for i in range(5):
        axes[i//2][0+i%2*2].plot(train_num_epoch, train_loss[i],  linestyle='-', color='b', label='Train Loss')
        axes[i//2][0+i%2*2].plot(test_num_epoch, test_loss[i],  linestyle='-', color='r', label='Test Loss')
        try:
            title = act_name[i]
        except:
            title=""
        axes[i//2][0+i%2*2].set_title(f"{title}_Loss ")
        axes[i//2][0+i%2*2].set_xlabel('Epoch')
        axes[i//2][0+i%2*2].set_ylabel('Loss')
        axes[i//2][0+i%2*2].grid(True)
        # plt.savefig(f"{name}_loss.png")
        # plt.show()

        # 创建一个图表
        # plt.figure(figsize=(10, 6))
        axes[i//2][1+i%2*2].plot(train_num_epoch, train_acc[i], linestyle='-', color='b', label='Train accuracy')
        axes[i//2][1+i%2*2].plot(test_num_epoch, test_acc[i], linestyle='-', color='r', label='Test accuracy')
        axes[i//2][1+i%2*2].set_title(f'{title}_Accuracy ')
        axes[i//2][1+i%2*2].set_xlabel('Epoch')
        axes[i//2][1+i%2*2].set_ylabel(" Accuracy ")
        axes[i//2][1+i%2*2].grid(True)

        # Add legend to each subplot
        axes[i//2][0+i%2*2].legend()
        axes[i//2][1+i%2*2].legend()

    # Adjust layout
    plt.tight_layout()


    plt.savefig(osp.join(dir,"contrast_acc_loss.png"))

def plot_contrast_loss(dir):
    
    name_list = os.listdir(dir)
    name_list = [name for name in name_list if name.endswith('.json')]
    loss_name = [name.split('_')[5] for name in name_list]
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for name in name_list:
        path = osp.join(dir,name)
        with open(path,"r") as f:
            log = json.load(f)
            train_loss.append([log['train_loss'][i] for i in range(len(log['train_loss'])) if i%12==0])
            train_acc.append([log['train_acc'][i] for i in range(len(log['train_acc'])) if i%12==0])
            test_loss.append(log['test_loss'])
            test_acc.append(log['test_acc'])
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]

    fig, axes= plt.subplots(2, 4, figsize=(20, 12))

    # 创建一个图表
    # plt.figure(figsize=(10, 6))
    for i in range(4):
        axes[i//2][0+i%2*2].plot(train_num_epoch, train_loss[i],  linestyle='-', color='b', label='Train Loss')
        axes[i//2][0+i%2*2].plot(test_num_epoch, test_loss[i],  linestyle='-', color='r', label='Test Loss')
        try:
            title = loss_name[i]
        except:
            title=""
        axes[i//2][0+i%2*2].set_title(f"{title}_Loss ")
        axes[i//2][0+i%2*2].set_xlabel('Epoch')
        axes[i//2][0+i%2*2].set_ylabel('Loss')
        axes[i//2][0+i%2*2].grid(True)
        # plt.savefig(f"{name}_loss.png")
        # plt.show()

        # 创建一个图表
        # plt.figure(figsize=(10, 6))
        axes[i//2][1+i%2*2].plot(train_num_epoch, train_acc[i], linestyle='-', color='b', label='Train accuracy')
        axes[i//2][1+i%2*2].plot(test_num_epoch, test_acc[i], linestyle='-', color='r', label='Test accuracy')
        axes[i//2][1+i%2*2].set_title(f'{title}_Accuracy ')
        axes[i//2][1+i%2*2].set_xlabel('Epoch')
        axes[i//2][1+i%2*2].set_ylabel(" Accuracy ")
        axes[i//2][1+i%2*2].grid(True)

        # Add legend to each subplot
        axes[i//2][0+i%2*2].legend()
        axes[i//2][1+i%2*2].legend()

    # Adjust layout
    plt.tight_layout()


    plt.savefig(osp.join(dir,"contrast_acc_loss.png"))

def plot_hyparam(key:str):
    sta_config = {
                "learning_rate": 0.0001,
                "weight_decay": 0.001,
                "momentum": 0.9,
                "batch_size": 100,
                "max_epoch": 100,
                "disp_freq": 50,
                "test_epoch": 5
            }
    all_file = [file for file in os.listdir("hyperparam") if file.endswith(".json")]
    baseline = False
    key_log = []
    for file in all_file:
        # print(file)
        with open(osp.join("hyperparam",file),"r") as f:
            log = json.load(f)
            
            if log["config"]== sta_config and baseline == False:
                baseline = True
                key_log.append(log)
            elif log["config"][key] != sta_config[key]:
                key_log.append(log)


    print(len(key_log))
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(20, 32))
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]
    color_list = ["blue", "red", "green", "yellow", "black", "purple","pink"]
    for i in range(len(key_log)):
        log = key_log[i]
        # print(1200//log["config"]["batch_size"])
        freq = 1200//log["config"]["batch_size"]
        if 1200//log["config"]["batch_size"] == 0:
            continue
        train_loss = log['train_loss']
        train_loss = [train_loss[i] for i in range(len(train_loss)) if i%freq==0]
        test_loss = log['test_loss']
        train_acc = log['train_acc']
        train_acc = [train_acc[i] for i in range(len(train_acc)) if i%freq==0]
        test_acc = log['test_acc']
        # print(len(train_loss))
        # print(len(train_num_epoch))
        ax1.plot(train_num_epoch, train_loss,  linestyle='-', color=color_list[i], label=f"{log['config'][key]}({train_loss[-1]:.2f})")
        ax2.plot(test_num_epoch, test_loss,  linestyle='-', color=color_list[i], label=f"{log['config'][key]}({test_loss[-1]:.2f})")
        ax3.plot(train_num_epoch, train_acc, linestyle='-', color=color_list[i], label=f"{log['config'][key]}({train_acc[-1]:.2f})")
        ax4.plot(test_num_epoch, test_acc, linestyle='-', color=color_list[i], label=f"{log['config'][key]}({test_acc[-1]:.2f})")
        ax1.set_title(f'{key} Train Loss ')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(" Loss ")
        ax1.grid(True)
        ax2.set_title(f'{key} Test Loss ')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(" Loss ")
        ax2.grid(True)
        ax3.set_title(f'{key} Train Accuracy ')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel(" Accuracy ")
        ax3.grid(True)
        ax4.set_title(f'{key} Test Accuracy ')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel(" Accuracy ")
        ax4.grid(True)
    # Add legend to each subplot
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"./hyperparam/{key}_acc_loss.png")

    

def plot_hyparam_hs():
    key = "hidden size"
    file_list = [file for file in os.listdir("hs") if file.endswith(".json")]
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(20, 32))
    train_num_epoch = range(100)
    test_num_epoch = [i for i in range(100) if i%5==0]
    color_list = ["blue", "red", "green", "yellow", "black", "purple","pink","brown"]
    hs_list = [16,32,64,128,256,512,1024,2048]
    time_list = []
    for i in range(len(file_list)):
        with open(osp.join("hs",file_list[i]),"r") as f:
            log = json.load(f)
        # print(1200//log["config"]["batch_size"])
        freq = 1200//log["config"]["batch_size"]
        if 1200//log["config"]["batch_size"] == 0:
            continue
        train_loss = log['train_loss']
        train_loss = [train_loss[i] for i in range(len(train_loss)) if i%freq==0]
        test_loss = log['test_loss']
        train_acc = log['train_acc']
        train_acc = [train_acc[i] for i in range(len(train_acc)) if i%freq==0]
        test_acc = log['test_acc']
        time_list.append(log["time"])
        # print(len(train_loss))
        # print(len(train_num_epoch))
        ax1.plot(train_num_epoch, train_loss,  linestyle='-', color=color_list[i], label=f"{hs_list[i]}({train_loss[-1]:.2f})")
        ax2.plot(test_num_epoch, test_loss,  linestyle='-', color=color_list[i], label=f"{hs_list[i]}({test_loss[-1]:.2f})")
        ax3.plot(train_num_epoch, train_acc, linestyle='-', color=color_list[i], label=f"{hs_list[i]}({train_acc[-1]:.2f})")
        ax4.plot(test_num_epoch, test_acc, linestyle='-', color=color_list[i], label=f"{hs_list[i]}({test_acc[-1]:.2f})")
        ax1.set_title(f'{key} Train Loss ')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(" Loss ")
        ax1.grid(True)
        ax2.set_title(f'{key} Test Loss ')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(" Loss ")
        ax2.grid(True)
        ax3.set_title(f'{key} Train Accuracy ')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel(" Accuracy ")
        ax3.grid(True)
        ax4.set_title(f'{key} Test Accuracy ')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel(" Accuracy ")
        ax4.grid(True)
    # Add legend to each subplot
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"./hs/{key}_acc_loss.png")


    # 绘制图形
    plt.clf()
    fig = plt.figure(figsize=(20, 12))
    plt.plot(hs_list, time_list,linestyle='-',linewidth=2)

    # 设置横坐标和纵坐标的标签
    plt.xlabel('Hidden Size')
    plt.ylabel('Train Time(s)')
    plt.savefig(f"./hs/time.png")
# name_list = os.listdir("./")
# name_list = [name for name in name_list if name.endswith(".json")]
# for name in name_list:
#     print(name)
#     plot_one_experiment(name)

# plot_contrast("./fc1_relu1_fc2_HingeLoss_2023-10-07-09-10-22.json",
#               "./fc1_relu1_fc2_HingeLoss_2023-10-07-01-09-12.json")
# plot_contrast_activation("./two_layer_activation/")
# plot_contrast_loss("./two_layer_loss/")
# plot_hyparam("learning_rate")
# plot_hyparam("weight_decay")
# plot_hyparam("momentum")
# plot_hyparam("batch_size")
plot_hyparam_hs()