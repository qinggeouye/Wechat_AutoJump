import numpy as np
import os
import cv2


class JumpData:
    def __init__(self):
        self.data_dir = '/resource/data'  # 训练数据的路径
        self.name_list = []
        self.get_name_list()
        self.val_name_list = self.name_list[:200]
        self.train_name_list = self.name_list[200:]
        self.name_list_raw = None

    def get_name_list(self):
        for i in range(3, 10):
            dir_para = os.path.join(self.data_dir, 'exp_%02d' % i)
            this_name = os.listdir(dir_para)
            this_name = [os.path.join(dir_para, name) for name in this_name]
            self.name_list = self.name_list + this_name
        self.name_list_raw = self.name_list
        self.name_list = filter(lambda name: 'res' in name, self.name_list)
        self.name_list = list(self.name_list)

        def _name_checker(name):
            position = name.index('_res')
            img_name = name[:position] + '.png'
            if img_name in self.name_list_raw:
                return True
            else:
                return False

        self.name_list = list(filter(_name_checker, self.name_list))

    def next_batch(self, batch_size=8):
        batch_name = np.random.choice(self.train_name_list, batch_size)
        batch_para = {}
        for idx, name in enumerate(batch_name):
            position = name.index('_res')
            img_name = name[:position] + '.png'
            x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[
                                                                       name.index('_w_') + 3: name.index('_w_') + 6]
            x, y = int(x), int(y)
            img = cv2.imread(img_name)
            img = img[320: -320, :, :]
            label = np.array([x, y], dtype=np.float32)
            mask1 = (img[:, :, 0] == 245)
            mask2 = (img[:, :, 1] == 245)
            mask3 = (img[:, :, 2] == 245)
            mask = mask1 * mask2 * mask3
            img[mask] = img[x - 320 + 10, y + 14, :]
            if idx == 0:
                batch_para['img'] = img[np.newaxis, :, :, :]
                batch_para['label'] = label.reshape([1, label.shape[0]])
            else:
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = label.reshape((1, label.shape[0]))
                batch_para['img'] = np.concatenate((batch_para['img'], img_tmp), axis=0)
                batch_para['label'] = np.concatenate((batch_para['label'], label_tmp), axis=0)
        return batch_para


if __name__ == '__main__':
    data = JumpData()
    batch = data.next_batch(1)
    print(batch['img'].shape)
