









recorded_accs = []

acc_avg, loss_avg = 0, 0
accs, losses = [], []
for inputs in data_module.train_dataloader():
    xs, ys = inputs
    preds = mg.model(xs)
    loss = torch.nn.functional.cross_entropy(preds, ys)
    acc = metric(preds, ys)
    accs.append(acc)
    losses.append(loss)
    if j > num_batchs:
        break
    j += 1
acc_avg = sum(accs) / len(accs)
loss_avg = sum(losses) / len(losses)
recorded_accs.append(acc_avg)