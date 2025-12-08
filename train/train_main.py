import numpy as np  # Numpy是一个强大的数学库，提供了大量数学函数和操作，尤其是处理大型多维数组和矩阵的时候好用
import matplotlib.pyplot as plt  # 可以创建静态、交互式和动画可视化的图库
from openpyxl import load_workbook  # 写入文件夹的

import torch  # 是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理领域
import torch.nn as nn  # 从torch库中导入nn模块，该模块包含了构建神经网络所需的类和函数
import torch.nn.functional as F  # 这个模块包括了构建神经网络时需要的函数式接口
import torch.optim as optim  # 包含了各种优化算法，用于神经网络的训练过程中更新权重

torch.set_default_tensor_type(
    "torch.DoubleTensor"
)  # 设置张量的默认类型为双精度浮点数，即64位浮点数
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # 导入这两个类，前者用于封装数据集，后者用于创建可迭代的数据加载器，可以批量加载数据，支持多线程加载
from torch.utils.data.dataset import (
    random_split,
)  # 导入这个函数，可将数据集随机分割为多个子集

import os
import sys

# 添加项目根目录到Python路径，确保能导入src下的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

import utils
import mlmodel
import time

if sys.platform == "win32":  # win32是windows平台的标识
    NUM_WORKERS = 0  # Windows does not support multiprocessing
else:
    NUM_WORKERS = 2
print("running on " + sys.platform + ", setting " + str(NUM_WORKERS) + " workers")

# 记录开始时间
start_time = time.time()

# Load the data and create some simple visualizations
for my_a in range(3, 4):
    print("dim_a = {}".format(my_a))
    dim_a = my_a
    features = ["v", "q", "pwm"]  # 定义一个列表，包含三个特征名称
    label = "fa"
    # label = 'my_fa'

    # Training data collected from the neural-fly drone
    dataset = "neural-fly"
    # 使用项目根目录的绝对路径
    dataset_folder = os.path.join(project_root, "data", "training")
    hover_pwm_ratio = 1.0

    # # Training data collected from an intel aero drone
    # dataset = 'neural-fly-transfer'
    # dataset_folder = os.path.join(project_root, 'data', 'training-transfer')
    # hover_pwm = 910 # mean hover pwm for neural-fly drone
    # intel_hover_pwm = 1675 # mean hover pwm for intel-aero drone
    # hover_pwm_ratio = hover_pwm / intel_hover_pwm # scaling ratio from system id

    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"  # 'intel-aero_fa-num-Tsp_v-q-pwm' # 使用f-string格式化字符串定义变量

    print(f"加载训练数据从: {dataset_folder}")
    RawData = utils.load_data(dataset_folder)
    Data = utils.format_data(RawData, features=features, output=label)

    testdata_folder = os.path.join(project_root, "data", "experiment")
    print(f"加载测试数据从: {testdata_folder}")
    RawData = utils.load_data(testdata_folder, expnames="(baseline_)([0-9]*|no)wind")
    TestData = utils.format_data(
        RawData, features=features, output=label, hover_pwm_ratio=hover_pwm_ratio
    )  # wind condition label, C, will not make sense for this data - that's okay since C is only used in the training process

    # for data in Data:
    #     utils.plot_subdataset(data, features, title_prefix="(Training data)")
    #
    # for data in TestData:
    #     utils.plot_subdataset(data, features, title_prefix="(Testing Data)")
    #     print(data)

    # Initialize some other hyperparameters
    options = {}
    options["dim_x"] = Data[0].X.shape[1]
    options["dim_y"] = Data[0].Y.shape[1]
    options["num_c"] = len(Data)
    print("dims of (x, y) are", (options["dim_x"], options["dim_y"]))
    print("there are " + str(options["num_c"]) + " different conditions")
    print(options)

    # Set hyperparameters
    options["features"] = features
    options["dim_a"] = dim_a
    options["loss_type"] = "crossentropy-loss"

    options["shuffle"] = True  # True: shuffle trajectories to data points
    options["K_shot"] = 32  # number of K-shot for least square on a
    options["phi_shot"] = 256  # batch size for training phi

    options["alpha"] = 0.01  # adversarial regularization loss
    options["learning_rate"] = 5e-4
    options["frequency_h"] = (
        2  # how many times phi is updated between h updates, on average
    )
    options["SN"] = 2.0  # maximum single layer spectral norm of phi
    options["gamma"] = 10.0  # max 2-norm of a
    options["num_epochs"] = 500

    # Dataset Generation
    # Trainset = []
    # Adaptset = []
    Trainloader = []
    Adaptloader = []
    for i in range(options["num_c"]):
        fullset = mlmodel.MyDataset(Data[i].X, Data[i].Y, Data[i].C)

        l = len(Data[i].X)
        if options["shuffle"]:
            trainset, adaptset = random_split(
                fullset, [int(2 / 3 * l), l - int(2 / 3 * l)]
            )
        else:
            trainset = mlmodel.MyDataset(
                Data[i].X[: int(2 / 3 * l)], Data[i].Y[: int(2 / 3 * l)], Data[i].C
            )
            adaptset = mlmodel.MyDataset(
                Data[i].X[int(2 / 3 * l) :], Data[i].Y[int(2 / 3 * l) :], Data[i].C
            )

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=options["phi_shot"],
            shuffle=options["shuffle"],
            num_workers=NUM_WORKERS,
        )
        adaptloader = torch.utils.data.DataLoader(
            adaptset,
            batch_size=options["K_shot"],
            shuffle=options["shuffle"],
            num_workers=NUM_WORKERS,
        )

        # Trainset.append(trainset)
        # Adaptset.append(adaptset)
        Trainloader.append(trainloader)  # for training phi
        Adaptloader.append(adaptloader)  # for LS on a

    # Domain Adversarially Invariant Meta Learning
    # Initialize the models
    # Store the model class definition in an external file so they can be referenced outside this script
    phi_net = mlmodel.Phi_Net(options)
    h_net = mlmodel.H_Net_CrossEntropy(options)

    criterion = nn.MSELoss()
    criterion_h = nn.CrossEntropyLoss()
    optimizer_h = optim.Adam(h_net.parameters(), lr=options["learning_rate"])
    optimizer_phi = optim.Adam(phi_net.parameters(), lr=options["learning_rate"])

    # Meta-Training Algorithm
    model_save_freq = 50  # How often to save the model

    # Create some arrays to save training statistics
    Loss_f = []  # combined force prediction loss
    Loss_c = []  # combined adversarial loss

    # Loss for each subdataset
    Loss_test_nominal = []  # loss without any learning
    Loss_test_mean = []  # loss with mean predictor
    Loss_test_phi = []  # loss with NN
    for i in range(len(TestData)):
        Loss_test_nominal.append([])
        Loss_test_mean.append([])
        Loss_test_phi.append([])

    # Training!
    for epoch in range(options["num_epochs"]):
        # Randomize the order in which we train over the subdatasets
        arr = np.arange(options["num_c"])
        np.random.shuffle(arr)

        # Running loss over all subdatasets
        running_loss_f = 0.0
        running_loss_c = 0.0

        for i in arr:
            with torch.no_grad():
                adaptloader = Adaptloader[i]
                kshot_data = next(iter(adaptloader))
                trainloader = Trainloader[i]
                data = next(iter(trainloader))

            optimizer_phi.zero_grad()

            """
            Least-square to get $a$ from K-shot data
            """
            X = kshot_data["input"]  # K x dim_x
            Y = kshot_data["output"]  # K x dim_y
            Phi = phi_net(X)  # K x dim_a
            Phi_T = Phi.transpose(0, 1)  # dim_a x K
            A = torch.inverse(torch.mm(Phi_T, Phi))  # dim_a x dim_a
            a = torch.mm(torch.mm(A, Phi_T), Y)  # dim_a x dim_y
            if torch.norm(a, "fro") > options["gamma"]:
                a = a / torch.norm(a, "fro") * options["gamma"]
            # print("第{}轮的第")

            """
            Batch training \phi_net
            """
            inputs = data["input"]  # B x dim_x
            labels = data["output"]  # B x dim_y

            c_labels = data["c"].type(torch.long)

            # forward + backward + optimize
            outputs = torch.mm(phi_net(inputs), a)
            loss_f = criterion(outputs, labels)
            temp = phi_net(inputs)

            loss_c = criterion_h(h_net(temp), c_labels)

            loss_phi = loss_f - options["alpha"] * loss_c
            loss_phi.backward()
            optimizer_phi.step()

            """
            Discriminator training
            """
            if np.random.rand() <= 1.0 / options["frequency_h"]:
                optimizer_h.zero_grad()
                temp = phi_net(inputs)

                loss_c = criterion_h(h_net(temp), c_labels)

                loss_h = loss_c
                loss_h.backward()
                optimizer_h.step()

            """
            Spectral normalization
            """
            if options["SN"] > 0:
                for param in phi_net.parameters():
                    M = param.detach().numpy()
                    if M.ndim > 1:
                        s = np.linalg.norm(M, 2)
                        if s > options["SN"]:
                            param.data = param / s * options["SN"]

            running_loss_f += loss_f.item()
            running_loss_c += loss_c.item()

        # Save statistics
        Loss_f.append(running_loss_f / options["num_c"])
        Loss_c.append(running_loss_c / options["num_c"])
        if epoch % 10 == 0:
            print(
                "[%d] loss_f: %.2f loss_c: %.2f"
                % (
                    epoch + 1,
                    running_loss_f / options["num_c"],
                    running_loss_c / options["num_c"],
                )
            )

            # # 打开文件（如果文件不存在会自动创建），并以写入模式打开
            # with open("a_{}_model.txt".format(dim_a), "a", encoding="utf-8") as file:
            #     # 写入数据
            #     file.write('[%d] loss_f: %.2f loss_c: %.2f' % (
            #         epoch + 1, running_loss_f / options['num_c'], running_loss_c / options['num_c']) + '\n')

        with torch.no_grad():
            for j in range(len(TestData)):
                loss_nominal, loss_mean, loss_phi = mlmodel.error_statistics(
                    TestData[j].X, TestData[j].Y, phi_net, h_net, options=options
                )
                Loss_test_nominal[j].append(loss_nominal)
                Loss_test_mean[j].append(loss_mean)
                Loss_test_phi[j].append(loss_phi)

        if epoch % model_save_freq == 0:
            # mlmodel.save_model(phi_net=phi_net, h_net=h_net, modelname=modelname + '-epoch-' + str(epoch) + 'myfadata', options=options)
            mlmodel.save_model(
                phi_net=phi_net,
                h_net=h_net,
                modelname=modelname + "-epoch-" + str(epoch),
                options=options,
            )

    # plot
    plt.subplot(2, 1, 1)
    plt.plot(Loss_f)
    plt.xlabel("epoch")
    plt.ylabel("f-loss [N]")
    plt.title("training f loss")
    plt.subplot(2, 1, 2)
    plt.plot(Loss_c)
    plt.title("training c loss")
    plt.xlabel("epoch")
    plt.ylabel("c-loss")
    plt.tight_layout()
    plt.show()

    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time} 秒")

    #
    for j in range(len(TestData)):
        plt.figure()
        # plt.plot(Loss_test_nominal[j], label='nominal')
        plt.plot(Loss_test_mean[j], label="mean")
        plt.plot(np.array(Loss_test_phi[j]), label="phi*a")
        # plt.plot(np.array(Loss_test_exp_forgetting[j]), label='exp forgetting')
        plt.legend()
        plt.title(f'Test data set {j} - {TestData[j].meta["condition"]}')

    plt.show()
    #
    # Choose final model
    stopping_epoch = 200
    options["num_epochs"] = stopping_epoch
    final_model = mlmodel.load_model(
        modelname=modelname + "-epoch-" + str(stopping_epoch)
    )

    # Error Analysis
    for i, data in enumerate(Data):
        print("------------------------------")
        print(data.meta["condition"] + ":")
        # mlmodel.vis_validation(t=data.meta['t'], x=data.X, y=data.Y, phi_net=final_model.phi, h_net=final_model.h, idx_adapt_start=0, idx_adapt_end=1000, idx_val_start=1000, idx_val_end=2000, c=Data[i].C, options=options)

    for data in Data:
        error_1, error_2, error_3 = mlmodel.error_statistics(
            data.X, data.Y, final_model.phi, final_model.h, options=options
        )
        print("**** c =", str(data.C), ":", data.meta["condition"], "****")
        print(f"Before learning: MSE is {error_1: .2f}")
        print(f"Mean predictor: MSE is {error_2: .2f}")
        print(f"After learning phi(x): MSE is {error_3: .2f}")
        print("")

        # 打开文件（如果文件不存在会自动创建），并以写入模式打开
        with open("a_{}_model_MSE.txt".format(dim_a), "a", encoding="utf-8") as file:
            # 写入数据
            file.write("train Data" + "\n")
            file.write("**** c =" + str(data.C) + ":" + data.meta["condition"] + "****")
            file.write(f"Before learning: MSE is {error_1: .2f}" + "\n")
            file.write(f"Mean predictor: MSE is {error_2: .2f}" + "\n")
            file.write(f"After learning phi(x): MSE is {error_3: .2f}" + "\n")
    file.close()

    # Test Data Error Analysis
    for i, data in enumerate(TestData):
        print("------------------------------")
        print(data.meta["condition"] + ":")
        # mlmodel.vis_validation(t=data.meta['t'], x=data.X, y=data.Y, phi_net=final_model.phi, h_net=final_model.h, idx_adapt_start=0, idx_adapt_end=1000, idx_val_start=1000, idx_val_end=2000, c=Data[i].C, options=options)

    for data in TestData:
        error_1, error_2, error_3 = mlmodel.error_statistics(
            data.X, data.Y, final_model.phi, final_model.h, options=options
        )
        print("**** :", data.meta["condition"], "****")
        print(f"Before learning: MSE is {error_1: .2f}")
        print(f"Mean predictor: MSE is {error_2: .2f}")
        print(f"After learning phi(x): MSE is {error_3: .2f}")
        print("")

        # # 写入excel文件
        # # 打开已存在的 Excel 文件
        # file_path = "a_test_data.xlsx"
        # wb = load_workbook(file_path)
        # ws = wb.active  # 获取当前活动的工作表
        # ws.cell(row=2, column=dim_a, value=dim_a)
        # # 判断是哪个风况
        # if data.meta['condition'] == '100wind':
        #     # 定义要写入的数字和目标行列
        #     row = 4  # 行号（从 1 开始计数）
        #     col = dim_a  # 列号（从 1 开始计数）
        #     ws.cell(row=row, column=col, value=error_1)
        #     row = 5
        #     ws.cell(row=row, column=col, value=error_2.item())
        #     row = 6
        #     ws.cell(row=row, column=col, value=error_3)
        #
        # elif data.meta['condition'] == '35wind':
        #     # 定义要写入的数字和目标行列
        #     row = 9  # 行号（从 1 开始计数）
        #     col = dim_a  # 列号（从 1 开始计数）
        #     ws.cell(row=row, column=col, value=error_1)
        #     row = 10
        #     ws.cell(row=row, column=col, value=error_2.item())
        #     row = 11
        #     ws.cell(row=row, column=col, value=error_3)
        # elif data.meta['condition'] == '70wind':
        #     # 定义要写入的数字和目标行列
        #     row = 14  # 行号（从 1 开始计数）
        #     col = dim_a  # 列号（从 1 开始计数）
        #     ws.cell(row=row, column=col, value=error_1)
        #     row = 15
        #     ws.cell(row=row, column=col, value=error_2.item())
        #     row = 16
        #     ws.cell(row=row, column=col, value=error_3)
        # elif data.meta['condition'] == 'nowind':
        #     # 定义要写入的数字和目标行列
        #     row = 19  # 行号（从 1 开始计数）
        #     col = dim_a  # 列号（从 1 开始计数）
        #     ws.cell(row=row, column=col, value=error_1)
        #     row = 20
        #     ws.cell(row=row, column=col, value=error_2.item())
        #     row = 21
        #     ws.cell(row=row, column=col, value=error_3)
        # # 保存修改后的 Excel 文件
        # wb.save(file_path)
        # print(f"MSE已成功追加到文件 {file_path} 。")

        # # 打开文件（如果文件不存在会自动创建），并以写入模式打开
        # with open("a_{}_model_MSE.txt".format(dim_a), "a", encoding="utf-8") as file:
        # # 写入数据
        #     file.write('test Data' + '\n')
        #     file.write('**** :'+data.meta['condition'] + '****' + '\n' )
        #     file.write(f'Before learning: MSE is {error_1: .2f}' + '\n')
        #     file.write(f'Mean predictor: MSE is {error_2: .2f}' + '\n')
        #     file.write(f'After learning phi(x): MSE is {error_3: .2f}' + '\n')

    # file.close()
