# Argparse here for training

from model import SparseUnet

if __name__ == '__main__':
    model = SparseUnet(shape=(512, 512, 1))

    model.train(train_dir='benchmark_EM/train',
                test_dir='benchmark_EM/test',
                out_dir='FINAL_512x512x4_200_epochs_only_sparse_run_3',
                epochs=200,
                batch_size=4,
                dense=False)

    # model.load('sparse_unet_epoch_118_val_loss_0.0000.hdf5')
    model.load('sparse_unet_epoch_193_val_loss_0.0000.hdf5')

    # img = tifffile.imread('../Data/CLEM_001/EM_001_50x50x50nm.tif')[80:90]

    img_lst = sorted(glob.glob(os.path.join('../Data/CLEM_003/EM04422_04_PNG', '*')))

    # img_pred = np.empty(img.shape)

    for i in tqdm(range(len(img_lst))):
        img_pred = model.predict(imageio.imread(img_lst[i]), tile_shape=(1167, 1167)) #(1167, 1167)

        imageio.imwrite(os.path.join('../Data/CLEM_003/EM0442_04_PNG_pred', 'pred_{}.png'.format(i)), img_pred.astype('float'))


    # tifffile.imwrite('EM_001_50x50x50nm_pred.tif', img_pred.astype('float32'))
    # img = imageio.imread('img.088.png')
    # img_pred = model.predict(img, tile_shape=(1167, 1167))
    # imageio.imwrite('img.088_pred.png', (img_pred > 0.8).astype('float'))
