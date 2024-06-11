#模型训练
import torch 
from tensorboardX import SummaryWriter
from Network import Retinex_Net, DConv, ResidualBlock,RI_BLOCK
from utils import list_images, get_image, load_dataset, get_train_images_auto
import random
from loss import loss_RT

def model_train(train_path,test_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_c = 1 # 1 - gray; 3 - RGB
    if in_c == 1:
        img_model = 'L'
    else:
        img_model = 'RGB'
    epoch = 200
    batch_size = 16
    train_num = 450

    #数据准备
    #训练集
    train_imgs_path = list_images(train_path)
    train_imgs_path = train_imgs_path[:train_num]
    random.shuffle(train_imgs_path)
    image_set_train, batches = load_dataset(train_imgs_path, batch_size)
    
    Model = Retinex_Net()

    #优化器
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)

    #添加到tensorboard
    writer = SummaryWriter('logs')
    #模型训练
    total_train_step = 0
    total_test_step = 0
    #训练开始
    Model.train()
    for i in range(epoch):
        print("-----------第{}轮训练开始-----------".format(i+1))
        for batch in range(batches):
            image_paths = image_set_train[batch * batch_size:(batch * batch_size + batch_size)]
            img = get_train_images_auto(image_paths,mode=img_model)
            #forward
            img = img.to(device)
            I,R = Model(img)

            #损失函数
            loss = loss_RT(img,I,R)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #display loss
            total_train_step += 1
            #print("训练次数:{}  loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

        #测试模型
        #数据准备
        #测试集
        test_num = 50
        test_imgs_path = list_images(test_path)
        test_imgs_path = test_imgs_path[:test_num]
        random.shuffle(test_imgs_path)
        image_set_test, batches = load_dataset(test_imgs_path, batch_size)

        Model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in range(batches):
                image_paths = image_set_test[batch * batch_size:(batch * batch_size + batch_size)]
                img = get_train_images_auto(image_paths,mode=img_model)
                #forward
                img = img.to(device)
                I,R = Model(img)

                #损失函数
                loss = loss_RT(img,I,R)
                total_test_loss += loss.item()
        print("整体测试集上loss:{}".format(total_test_loss/3))
        writer.add_scalar("test_loss",total_test_loss/3,i)
        if i+1 >= 250:
            #模型保存
            path = "./" + "model{}.pth".format(i+1)
            torch.save(Model,path)

def fusion_train(h_train_path,h_test_path,l_train_path,l_test_path):
    in_c = 1 # 1 - gray; 3 - RGB
    if in_c == 1:
        img_model = 'L'
    else:
        img_model = 'RGB'
    epoch = 400
    train_num = 450
    test_num = 50
    #导入与训练的Retinex模型
    Retinex = torch.load('../input/predata/pre/model/model400.pth')
    #训练集
    h_train_imgs_path = list_images(h_train_path)
    h_train_imgs_path = h_train_imgs_path[:train_num]
    l_train_imgs_path = list_images(l_train_path)
    l_train_imgs_path = l_train_imgs_path[:train_num]
    #测试集
    h_test_imgs_path = list_images(h_test_path)
    h_test_imgs_path = h_test_imgs_path[:test_num]
    l_test_imgs_path = list_images(l_test_path)
    l_test_imgs_path = l_test_imgs_path[:test_num]
    
    Model = RI_BLOCK()
    Model = Model.to(device)
    
    #优化器
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.0001)
    
    #模型训练
    total_train_step = 0
    total_test_step = 0
    #添加到tensorboard
    writer = SummaryWriter('logs')
    #训练开始
    Model.train()
    for i in range(epoch):
        print("-----------第{}轮训练开始-----------".format(i+1))
        random.shuffle(h_train_imgs_path)
        random.shuffle(l_train_imgs_path)
        h_image_set_train, batches = load_dataset(h_train_imgs_path, batch_size)
        l_image_set_train, batches = load_dataset(l_train_imgs_path, batch_size)
        for batch in range(batches):
            h_image_paths = h_image_set_train[batch * batch_size:(batch * batch_size + batch_size)]
            h_img = get_train_images_auto(h_image_paths,mode=img_model)
            l_image_paths = l_image_set_train[batch * batch_size:(batch * batch_size + batch_size)]
            l_img = get_train_images_auto(l_image_paths,mode=img_model)
            #forward
            h_img = h_img.to(device)
            l_img = l_img.to(device)
            
            HI,HR = Retinex(h_img)
            LI,LR = Retinex(l_img)
            
            R = torch.cat([HR,LR],1)
            I = torch.cat([HI,LI],1)
            
            S = Model(R,I)

            #损失函数
            loss = loss_Fusion(S,h_img)
            
            #backward
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            
            #display loss
            total_train_step += 1
            print("训练次数:{}  loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
           

        Model.eval()
        h_image_set_test, batches = load_dataset(h_test_imgs_path, batch_size)
        l_image_set_test, batches = load_dataset(l_test_imgs_path, batch_size)
        total_test_loss = 0
        with torch.no_grad():
            for batch in range(batches):
                h_image_paths = h_image_set_test[batch * batch_size:(batch * batch_size + batch_size)]
                h_img = get_train_images_auto(h_image_paths,mode=img_model)
                l_image_paths = l_image_set_test[batch * batch_size:(batch * batch_size + batch_size)]
                l_img = get_train_images_auto(l_image_paths,mode=img_model)
                
                #forward
                h_img = h_img.to(device)
                l_img = l_img.to(device)

                HI,HR = Retinex(h_img)
                LI,LR = Retinex(l_img)

                R = torch.cat([HR,LR],1)
                I = torch.cat([HI,LI],1)

                S = Model(R,I)

                #损失函数
                loss = loss_Fusion(S,h_img)
                total_test_loss += loss.item()
        print("整体测试集上loss:{}".format(total_test_loss/batches))
        writer.add_scalar("test_loss",total_test_loss/batches,i)
        if i+1 >= 250:
            #模型保存
            path = "./" + "model{}.pth".format(i+1)
            torch.save(Model,path)