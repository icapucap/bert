import time
import torch




def run_train(config, model, train_loader, eval_loader, writer):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=config.lr)
    
    training_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    for epoch in range(config.epochs):

        model.train()

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.epochs))
        start_time = time.time()

        total_loss = 0

        for batch_no, batch in enumerate(train_loader):

            source = batch[0].to(device)
            target = batch[1].to(device)

            model.zero_grad()        

            loss, logits = model(source, target)
            total_loss += loss.item()
        
            logits = logits.detach().cpu().numpy()
            label_ids = target.to('cpu').numpy()

            loss.backward()

            optimizer.step()
            scheduler.step()

        #Logging the loss and accuracy (below) in Tensorboard
        avg_train_loss = total_loss / len(train_loader)            
        training_loss_values.append(avg_train_loss)

        for name, weights in model.named_parameters():
            writer.add_histogram(name, weights, epoch)

        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Running Validation...")

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0

        for batch_no, batch in enumerate(eval_loader):
        
            source = batch[0].to(device)
            target = batch[1].to(device)
        
            with torch.no_grad():        
                loss, logits = model(source, target)

            logits = logits.detach().cpu().numpy()
            label_ids = target.to('cpu').numpy()
        
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            eval_loss += loss

            nb_eval_steps += 1

        avg_valid_acc = eval_accuracy/nb_eval_steps
        avg_valid_loss = eval_loss/nb_eval_steps
        validation_loss_values.append(avg_valid_loss)
        validation_accuracy_values.append(avg_valid_acc)

        writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
        writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
        writer.flush()

        print("Avg Val Accuracy: {0:.2f}".format(avg_valid_acc))
        print("Average Val Loss: {0:.2f}".format(avg_valid_loss))
        print("Time taken by epoch: {0:.2f}".format(time.time() - start_time))

    return training_loss_values, validation_loss_values, validation_accuracy_values


