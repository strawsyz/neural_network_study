epochs = 35          # 训练的轮数
batch_size = 40      # 一次训练的batch_size条数据


def train_model(model,X_train, Y_train, X_val, Y_val, epochs):
    model.fit(X_train, Y_train, batch_size = batch_size, 
        epochs = epochs, validation_data=(X_val, Y_val))
    # 保存训练好的模型
    model.save_weights('model_weights.h5', overwrite=True)  
    return model  