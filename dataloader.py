# DataLoader
class SparseDataGenerator(Sequence):

    def __init__(self,
                 data_dir,
                 batch_size=32,
                 shape=(256,256,1),
                 dense=False,
                 augment=True):

        self.dense = dense
        self.augment = augment

        # Training Directory
        img_list = glob.glob(os.path.join(data_dir, 'images/*.png'))
        img_list.sort()

        # Mask directory contains three sub-directories each containing
        # binary mask corresponding to foreground, background and None
        if self.dense:
            msk_list = glob.glob(os.path.join(data_dir, 'masks/*.png'))
            msk_list.sort()

            self.data_list = list(zip(img_list,
                                      msk_list))
        else:
            msk_bg_list = glob.glob(os.path.join(data_dir, 'masks/background/*.png'))
            msk_fg_list = glob.glob(os.path.join(data_dir, 'masks/foreground/*.png'))

            msk_bg_list.sort()
            msk_fg_list.sort()

            self.data_list = list(zip(img_list,
                                      msk_bg_list,
                                      msk_fg_list))

        self.on_epoch_end()
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        if self.augment:
            return 2 * len(self.data_list) // self.batch_size
        else:
            return len(self.data_list) // self.batch_size

    def __getitem__(self, idx):

        # Instantiate batches
        img_batch = np.zeros((self.batch_size,
                              self.shape[0],
                              self.shape[1],
                              self.shape[2]))
        msk_batch = np.zeros((self.batch_size,
                              self.shape[0],
                              self.shape[1],
                              3))

        # If densely annotated, background layer is 1-foreground and None class is 1-bg-fg in all cases
        k = idx % (len(self.data_list) // self.batch_size)
        for i in range(k * self.batch_size , k * self.batch_size + self.batch_size):

            j = i % self.batch_size

            img_batch[j, :, :, 0] = self._imread(self.data_list[i][0],
                                                 (self.shape[0], self.shape[1]),
                                                 'float')
            if self.dense:
                msk_batch[j, :, :, 1] = self._imread(self.data_list[i][1],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)
                msk_batch[j, :, :, 0] = np.ones((self.shape[0], self.shape[1])) - msk_batch[j, :, :, 1]
            else:
                msk_batch[j, :, :, 0] = self._imread(self.data_list[i][1],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)
                msk_batch[j, :, :, 1] = self._imread(self.data_list[i][2],
                                                     (self.shape[0], self.shape[1]),
                                                     'float',
                                                     binary=True)

            msk_batch[j, :, :, 2] = np.ones((self.shape[0], self.shape[1])) - msk_batch[j, :, :, 1] - msk_batch[j, :, :, 0]

        if self.augment:

            # Roughly augment 50% of batches
            rand = np.random.random()

            if rand > 0.5:
                return img_batch, msk_batch

            else:
                seq = iaa.Sequential([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Flipud(0.5), # vertical flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    iaa.Sometimes(
                        0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.LinearContrast((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)),
                    iaa.Multiply((0.8, 1.2)),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-25, 25),
                        shear=(-8, 8)
                    )
                ], random_order=True) # apply augmenters in random order

                # For augmentation purposes, collapse segmentation masks from 4D to 3D
                msk_batch_temp = np.zeros((msk_batch.shape[0],
                                           msk_batch.shape[1],
                                           msk_batch.shape[2]), dtype=np.int32)
                msk_batch_temp[np.where(msk_batch[:,:,:,0] == 1)] = 0
                msk_batch_temp[np.where(msk_batch[:,:,:,1] == 1)] = 1
                msk_batch_temp[np.where(msk_batch[:,:,:,2] == 1)] = 2
                msk_batch_temp = np.expand_dims(msk_batch_temp, axis=3)

                # msk_batch_temp = SegmentationMapsOnImage(msk_batch_temp, shape=(msk_batch.shape[1], msk_batch.shape[2], 1))
                # Returns a list of images --> Must repopulate an array
                img_aug_list, msk_aug_list = seq(images=(img_batch*255).astype(np.uint8),
                                                 segmentation_maps=msk_batch_temp)

                # Repopulate img array
                img_batch_aug = np.zeros(img_batch.shape)
                for i in range(len(img_aug_list)):
                    img_batch_aug[i] = img_aug_list[i]

                # Re-expand from collapsed mask to one-hot-encoded mask
                msk_batch_aug = np.zeros((msk_batch.shape[0],
                                          msk_batch.shape[1],
                                          msk_batch.shape[2],
                                          3))

                # Assign entire collapsed mask to one channel and then set all but desired value to zero
                for i, msk in enumerate(msk_aug_list):
                    msk = np.squeeze(msk)
                    msk_temp_zero = np.zeros(msk.shape)
                    msk_temp_zero[np.where(msk == 0)] = 1
                    msk_temp_one = np.zeros(msk.shape)
                    msk_temp_one[np.where(msk == 1)] = 1
                    msk_temp_two = np.zeros(msk.shape)
                    msk_temp_two[np.where(msk == 2)] = 1
                    msk_batch_aug[i,:,:,0] = msk_temp_zero
                    msk_batch_aug[i,:,:,1] = msk_temp_one
                    msk_batch_aug[i,:,:,2] = msk_temp_two

                img_batch_aug = (img_batch_aug/255).astype(np.float64)

                return img_batch_aug, msk_batch_aug

        else:
            return img_batch, msk_batch


    def on_epoch_end(self):

        random.shuffle(self.data_list)

    def batch_weights(self):
        # Compute from sample of 100 masks --> must ensure that None class is not counted...
        # Weight of class zero is all zero masks added divided by all
        return (1, 3, 0)

    def _imread(self, img_path, size, dtype, binary=False):

        try:
            img = imageio.imread(img_path, pilmode='L').astype(np.float)

            resized_img = transform.resize(img, size, mode='constant', preserve_range=True)
            out_img = exposure.rescale_intensity(resized_img, out_range=dtype)

            if binary:
                return np.squeeze(out_img) > 0
            else:
                return color.rgb2gray(out_img)
        except ValueError:
            print('Corrupted filename: ', img_path)
            return np.zeros(size)
