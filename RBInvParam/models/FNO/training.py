import os
import torch
import time
from torch import nn
import pickle
#from Feature_Engineering.learn_min_max import custom_normalizer
#from utils.help_functions import save_checkpoint, load_checkpoint, create_mask
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader
from RBInvParam.models.FNO.FNO import FNO1d_new

def training(
    train_loader : DataLoader,
    val_loader : DataLoader,
    model : FNO1d_new,
    checkpoint_path: Path,
    start_epoch : int = 0,
    num_epoch : int = 10, 
):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Training parameters
    learning_rate = 1e-3
    #weight_decay = 1e-10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_history = []
    val_loss_history = []
    # relative_loss_train = []
    # relative_loss_val = []
    stats_list = []

    model.to(device)
    start = time.time()

    for epoch in range(start_epoch, start_epoch+num_epoch):
        print(f'Epoch: {epoch}')
        model.train()
    
        for train_batch_idx, (train_input, train_res) in (enumerate(train_loader)):
            train_pred = model(train_input)
            train_loss = loss_fn(train_pred, train_res)

            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()
            print(f"train_loss: {train_loss.item():>7f}")

            model.eval()
            num_batches = len(val_loader)
            with torch.no_grad():
                val_loss = 0
                for val_batch_idx, (val_input, val_res) in (enumerate(val_loader)):
                    val_pred = model(val_input)
                    val_loss += loss_fn(val_pred, val_res)
                val_loss /= num_batches
                val_loss_history.append(val_loss)
                print(f"Avg validation loss: {val_loss:>8f} \n")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, 
        checkpoint_path / 'checkpoint.pth'
    )

    endtime = (time.time() - start) / 60

def get_param_stats(model):
    stats = defaultdict(dict)
    for name, p in model.named_parameters():
        if p.requires_grad:
            w = p.data
            g = p.grad if p.grad is not None else torch.zeros_like(w)
            stats[name]['weight_norm'] = w.norm(p=2).item()
            stats[name]['grad_norm'] = g.norm(p=2).item()
            stats[name]['grad_to_weight_ratio'] = (g.norm(p=2) / w.norm()).item() + 1e-12

    return stats


# # Implements the function solely for training a pytorch neural network
# # should be abstract enough to work for several different architectures
# def training_(training_generator: torch.utils.data.DataLoader, 
#              validation_generator: torch.utils.data.DataLoader,
#              model: nn.Module, 
#              loss_fn, 
#              optimizer,
#              size: int, 
#              batch_size: int, 
#              scale_list: list, 
#              features, 
#              max_length: int, 
#              model_file: str,
#              loss_train, 
#              loss_val, 
#              relative_loss_train, 
#              relative_loss_val,
#              #scheduler,
#              stats_list,
#              start_epoch=0,
#              num_epoch=10,
#              ):

#     # GPU check
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda:0' if use_cuda else 'cpu')
#     print(f'GPU is used for training: {use_cuda}')
#     torch.backends.cudnn.benchmark = True

#     model.to(device)

#     # To protocol time
#     start = time.time()

#     # Training
#     for epoch in range(start_epoch, start_epoch+num_epoch):
#         print(f'Epoch: {epoch}')
#         model.train()
#         loss_batch = 0
#         relative_loss_batch = 0

#         for batch, (batch_hdf5, batch_xlsx, batch_basket, batch_hotheel, batch_res, sequence_length, _) in (
#                 enumerate(training_generator)):
#             batch_hdf5, batch_xlsx, batch_basket, batch_hotheel, batch_res = (batch_hdf5.to(device),
#                                                                               batch_xlsx.to(device),
#                                                                               batch_basket.to(device),
#                                                                               batch_hotheel.to(device),
#                                                                               batch_res.to(device))

#             # Scale
#             batch_hdf5, batch_xlsx, batch_basket, batch_hotheel, batch_res = custom_normalizer(scale_list, batch_hdf5,
#                                                                                                    batch_xlsx,
#                                                                                                    batch_basket,
#                                                                                                    batch_hotheel,
#                                                                                                    batch_res, features, device)


#             # Data is collected padded but need to multiply prediction with mask
#             mask = create_mask(batch_res.shape[0], max_length, len(features['res_hdf5']), sequence_length).to(device)

#             # Forward
#             pred = model(batch_hdf5, batch_xlsx, batch_basket, batch_hotheel)

#             # Loss
#             loss = loss_fn(pred, batch_res) * mask
#             loss_sum = loss.sum()
#             loss_mean = loss_sum / mask.sum()

#             # Metric
#             loss_sqrt = torch.sqrt(loss_sum)
#             batch_res_squared = (batch_res * mask) ** 2
#             batch_res_sum = batch_res_squared.sum()
#             batch_res_sqrt = torch.sqrt(batch_res_sum)
#             #relative_loss = (loss_sqrt / ((batch_res ** 2*mask).sum()/mask.sum())) ** 0.5 * 100
#             relative_loss = (loss_sqrt / batch_res_sqrt) * 100

#             # Protocol
#             loss_batch += loss_mean.item()
#             relative_loss_batch += relative_loss.item()

#             # Backpropagation
#             optimizer.zero_grad()
#             loss_mean.backward()

#             # Clip Gradient
#             #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()

#             # Display
#             loss, relative_loss, current = loss_mean.item(), relative_loss.item(), batch * batch_size + len(pred)
#             print(f"loss: {loss:>7f} relative loss: {relative_loss:>7f}  [{current:>5d}/{size:>5d}]")

#         loss_train.append(loss_batch/len(training_generator))
#         relative_loss_train.append(relative_loss_batch/len(training_generator))

#         stats_list.append(get_param_stats(model))

#         # Validation
#         model.eval()

#         num_batches = len(validation_generator)
#         test_loss = 0
#         test_relative_loss = 0

#         with torch.no_grad():
#             for inp_hdf5, inp_xlsx, inp_basket, inp_hotheel, res_hdf5, sequence_length, _ in validation_generator:
#                 inp_hdf5, inp_xlsx, inp_basket, inp_hotheel, res_hdf5 = (inp_hdf5.to(device), inp_xlsx.to(device),
#                                                                          inp_basket.to(device), inp_hotheel.to(device),
#                                                                          res_hdf5.to(device))

#                 # Scale
#                 inp_hdf5, inp_xlsx, inp_basket, inp_hotheel, res_hdf5 = (custom_normalizer(scale_list, inp_hdf5,
#                                                                                            inp_xlsx, inp_basket,
#                                                                                            inp_hotheel, res_hdf5,
#                                                                                            features, device))

#                 # Mask
#                 mask = create_mask(res_hdf5.shape[0], max_length, len(features['res_hdf5']), sequence_length).to(device)

#                 # Forward
#                 val_pred = model(inp_hdf5, inp_xlsx, inp_basket, inp_hotheel)

#                 # Loss
#                 loss = loss_fn(val_pred, res_hdf5) * mask
#                 loss_sum = loss.sum()
#                 loss_mean = loss_sum / mask.sum()

#                 # Metric
#                 loss_sqrt = torch.sqrt(loss_sum)
#                 batch_res_squared = (res_hdf5 * mask) ** 2
#                 batch_res_sum = batch_res_squared.sum()
#                 batch_res_sqrt = torch.sqrt(batch_res_sum)
#                 # relative_loss = (loss_sqrt / ((batch_res ** 2*mask).sum()/mask.sum())) ** 0.5 * 100
#                 relative_loss = (loss_sqrt / batch_res_sqrt) * 100

#                 test_loss += loss_mean.item()
#                 test_relative_loss += relative_loss.item()

#         test_loss /= num_batches
#         #scheduler.step(test_loss)
#         test_relative_loss /= num_batches

#         loss_val.append(test_loss)
#         relative_loss_val.append(test_relative_loss)
#         print(f"Avg validation loss: {test_loss:>8f} \n")
#         print(f"Avg validation relative loss: {test_relative_loss:>8f} \n")

#         # Save the model in path file according to the trained architecture
#         save_checkpoint(model, optimizer, epoch+1, model_file + 'checkpoint.pth')

#         # Save also the loss training and validation
#         with open(model_file + "loss_val", "wb") as fp:  # Pickling
#             pickle.dump(loss_val, fp)
#         with open(model_file + "loss_train", "wb") as fp:  # Pickling
#             pickle.dump(loss_train, fp)
#         with open(model_file + "relative_loss_val", "wb") as fp:  # Pickling
#             pickle.dump(relative_loss_val, fp)
#         with open(model_file + "relative_loss_train", "wb") as fp:  # Pickling
#             pickle.dump(relative_loss_train, fp)
#         with open(model_file + "stats_list", "wb") as fp:  # Pickling
#             pickle.dump(stats_list, fp)

#         endtime = (time.time() - start) / 60
#         with open(model_file + 'training_time.txt', 'w') as text_file:
#             text_file.write(f'Training time={endtime} min\n')
#         text_file.close()
#         print(f'Training and validation of epoch {epoch} took {endtime} minutes.\n')

#     with open(model_file + 'features.pkl', 'wb') as fp:
#         pickle.dump(features, fp)


# def batch_training(features, 
#                    training_generator, 
#                    validation_generator, 
#                    num_epoch, 
#                    size, 
#                    batch_size, 
#                    data_file,
#                    model_file, 
#                    model, 
#                    max_length: int,):

#     # Preliminaries regarding hardware
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda:0' if use_cuda else 'cpu')
#     torch.backends.cudnn.benchmark = True

#     # Scale the input with a previous learned min max range
#     with open(data_file + 'min_max_value_list', 'rb') as fp:
#             scale_list = pickle.load(fp)

#     # Training parameters
#     learning_rate = 1e-3
#     weight_decay = 1e-10
#     loss_fn = nn.MSELoss(reduction='none')
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Resume if checkpoint exists
#     start_epoch = 0
#     if os.path.exists(model_file + "checkpoint.pth"):
#         start_epoch = load_checkpoint(model, optimizer, device, model_file + 'checkpoint.pth')

#         with open(model_file + "loss_val", "rb") as fp:  # Pickling
#             loss_val = pickle.load(fp)
#         with open(model_file + "loss_train", "rb") as fp:  # Pickling
#             loss_train = pickle.load(fp)
#         with open(model_file + "relative_loss_val", "rb") as fp:  # Pickling
#             relative_loss_val = pickle.load(fp)
#         with open(model_file + "relative_loss_train", "rb") as fp:  # Pickling
#             relative_loss_train = pickle.load(fp)
#     else:
#         loss_train = []
#         loss_val = []
#         relative_loss_train = []
#         relative_loss_val = []

#     #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
#     stats_list = []

#     # Training
#     training(training_generator, model, loss_fn, optimizer, size, batch_size, validation_generator,
#              scale_list, features, max_length, model_file, loss_train, loss_val, relative_loss_train, relative_loss_val,
#              #scheduler,
#              stats_list,
#              start_epoch, num_epoch,)