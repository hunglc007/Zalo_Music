from data_parser import DataParser
from keras import backend as K
import os
from model import ResNet50
from keras.utils import plot_model
from keras import callbacks
import numpy as np

def generate_minibatches(dataParser, train=True):
    # pdb.set_trace()
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.training_ids, dataParser.batch_size_train)
        else:
            batch_ids = np.random.choice(dataParser.validation_ids, dataParser.batch_size_train)
        ims, ems = dataParser.get_batch(batch_ids)
        yield(ims, ems)
if __name__ == "__main__":
    # params
    model_name = 'ResnetMusic'
    model_dir     = os.path.join('checkpoints', model_name)
    csv_fn        = os.path.join(model_dir, 'full_continue_train_log.csv')
    checkpoint_fn = os.path.join(model_dir, 'full_continue_checkpoint.{epoch:02d}-{val_loss:.2f}.h5')

    batch_size_train = 16
    epochs = 100
    K.set_image_data_format('channels_last')
    K.image_data_format()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # prepare data
    dataParser = DataParser(batch_size_train)

    # model
    model = ResNet50()
    modelPath = 'checkpoints/ResnetMusic/29000_checkpoint.06-2.42.h5'
    # model.load_weights(modelPath)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    plot_model(model, to_file=os.path.join(model_dir, 'model.pdf'), show_shapes=True)

    # training
    # call backs
    checkpointer = callbacks.ModelCheckpoint(filepath=checkpoint_fn, verbose=1)
    csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')

    train_history = model.fit_generator(
                        generate_minibatches(dataParser,),
                        steps_per_epoch=dataParser.steps_per_epoch,  #batch size
                        epochs=epochs,
                        validation_data=generate_minibatches(dataParser, train=False),
                        validation_steps=dataParser.validation_steps,
                        callbacks=[checkpointer, csv_logger])

    print(train_history)