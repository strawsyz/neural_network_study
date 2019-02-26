

def test_model(model,X_test,Y_test):
    '''测试模型，输出测试误差'''
    model.load_weights('model_weights.h5')  
    score = model.evaluate(X_test, Y_test)
    return score  