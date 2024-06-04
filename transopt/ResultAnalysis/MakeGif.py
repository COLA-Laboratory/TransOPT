import os
from PIL import Image

def make_gif(folder_path):
    # 获取文件夹中的所有图片文件
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]

    # 按照图片序号进行排序
    image_files.sort(key=lambda x: int(x.split('_')[0]))

    images = []
    for file in image_files:
        # 读取每个图片文件
        image_path = os.path.join(folder_path, file)
        image = Image.open(image_path)

        # 将图片添加到列表中
        images.append(image)

    # 设置保存 GIF 的文件路径和名称
    gif_path = os.path.join(folder_path, 'animation.gif')

    # 将图片列表保存为 GIF 动画
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0,)

if __name__ == '__main__':
    task_list_2d = [
        'Ackley_10_s',
        # 'StyblinskiTang_10_s',
        'MPB5_10_s',
        'LevyR_10_s',
        # 'SVM_10_s',
    ]
    task_list_5d = [
        # 'Ackley_10_s',
        # 'StyblinskiTang_10_s',
        # 'MPB5_10_s',
        # 'LevyR_10_s',
        'NN_72_s',
    ]

    task_list_8d = [
        # 'Ackley_10_s',
        # 'StyblinskiTang_10_s',
        # 'MPB5_10_s',
        'LevyR_10_s',
        # 'XGB_10_s',
    ]

    Dim_ = 2
    Method_list = [
        # 'INC_MHGP',
        # 'WS_RGPE',
        # 'MT_MOGP',
        # 'LFL_MOGP',
        # 'ELLA_GP',
        # 'BO_GP',
        'TMTGP'
    ]
    # Seed_list = list(range(10))
    Seed_list = [0]

    Exp_name = 'test5'
    Exper_floder = '../../LFL_experiments/{}'.format(Exp_name)

    if Dim_ == 2:
        task_list = task_list_2d
    elif Dim_ == 5:
        task_list = task_list_5d
    elif Dim_ == 8:
        task_list = task_list_8d

    # for Method in Method_list:
    #     for Prob in task_list:
    #         for seed in Seed_list:
    #             for i in range(int(Prob.split('_')[1])):
    #                 make_gif(Exper_floder+f"/figs/contour/{Method}/{seed}/{Prob.split('_')[0]}_{i}_{Prob.split('_')[2]}")

    for Method in Method_list:
        for Prob in task_list:
            for seed in Seed_list:
                for i in range(int(Prob.split('_')[1])):
                    make_gif(Exper_floder+f"/figs/contour/{Method}/{seed}/{Prob.split('_')[0]}_{i}_{Prob.split('_')[2]}")