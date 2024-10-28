from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from livelossplot import PlotLossesKeras

def trainer(model,epochs,train_ds,val_ds):
    '''
    
    takes the model to be trained, epochs , train dataset and validation dataset and returns model fitted
    
    '''
    steps_per_epoch=train_ds.n//train_ds.batch_size
    validation_steps = val_ds.n//val_ds.batch_size
    
    checkpoint=ModelCheckpoint("model_weights.h5",monitor='val_accuracy',
                          save_weights_only=True,model='max',verbose=1)
    
    reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2, min_lr=0.0001,mode='auto')
    
    callbacks=[PlotLossesKeras(),checkpoint,reduce_lr]
    
    history = model.fit(
            x=train_ds,
             steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks)
    
    return history



    