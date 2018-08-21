���Ľ��������ʹ��kerasʵ��transfer learning��ʹ��cifar10���ݿ��vgg16ģ�͡���ν��Ǩ��ѧϰ�����ǽ�ĳ��ģ����ĳ�����ݿ��ϵ��������������������ȡ����Ӧ�õ���һ�����ݿ⣬��Щ�������������Ͼ���ģ�͵�Ȩ�أ��ô������ݿ���Ԥѵ����ģ�͵�Ȩ�س�ʼ����ģ�ͣ����Ը���ѵ������ģ�͡�

# һ����model zoo����imagenetԤѵ��ģ��
	
# ����ģ�����
������ȡ�㱣�ֲ��䣬�����������һ��ȫ���Ӳ�ĵ�Ԫ�����޸�Ϊ��ǰ���ݿ����������
	��ģ��ѵ��ʱ��ʹ��Ԥѵ��ģ����������ȡ���Ȩ�س�ʼ����ģ�Ͷ�Ӧ�Ĳ㡣
	
	base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    predictions = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
	
����small dataset����Ҫ����������ȡ���ͼ�������ȡ�㣺  
	
	for layer in base_model.layers:  
		layer.trainable = False
	
# ����ģ�ͱ���  
ָ���Ż�������ʧ����������������  

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
	
# �ġ�ģ��ѵ��
�˴���Ҫָ��epochs��batch_size��flow_from_directory��ָ����
	
	tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath)

    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        verbose=1,
                        callbacks=[tensorboard, checkpointer],
                        validation_data=test_generator,
                        validation_steps=len(test_generator),
                        initial_epoch=initial_epoch,
                        workers=4)
	
tensorboard���ӻ������  
*graphs*:
![Image text](resources/tf_graphs.png)
  
*scalars*:  
![Image text](resources/acc.jpg)
![Image text](resources/loss.jpg)  
![Image text](resources/val_acc.jpg)
![Image text](resources/val_loss.jpg)   

*plot_model*:
![Image text](resources/vgg16.png)  

# �塢ģ������
����ģ�͵�����   

	score_evaluate = model.evaluate_generator(generator=test_generator,
                                              steps=len(test_generator),
                                              workers=4)
	    
	