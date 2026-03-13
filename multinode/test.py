    #   * weight_decay: weight decay for the Adam optimizer
    #   * lr: learning rate for the Adam optimizer (2e-5 to 5e-5 recommended)
    #   * warmup_steps: number of warmup steps to (linearly) reach the set
    #     learning rate
    #
    # We also need to grab the training parameters from the pretrained model.
    
    num_epochs = 4
    weight_decay = 0.01
    lr = 2e-5
    warmup_steps = int(0.2 * len(train_loader))
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader) * num_epochs)
    
    # Training loop
    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, scheduler, optimizer)
        log_measures(train_ret, log, "train", epoch)
        
        val_ret = test(validation_loader, model)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: "
            f"train loss: {train_ret['loss']:.6f} "
            f"train accuracy: {train_ret['accuracy']:.2%}, "
            f"val accuracy: {val_ret['accuracy']:.2%}")
    
    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))
    
    # Inference
    ret = test(test_loader, model)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")
    
    
    if __name__ == "__main__":
        main()
